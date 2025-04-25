"""
Optimised CLI wrapper for multi‑GPU NRMS training.

Launch example:
    CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. torchrun --standalone --nproc_per_node=4 \
        cli/train_nrms_ddp.py --epochs 3 --batch 128 --amp --num_workers 4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random

import mlflow
import mlflow.system_metrics
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from config.paths import DATA_DIR, MLFLOW_EXPERIMENT
from data.preprocess import load_and_prepare_nrms
from datasets.nrms import NRMSDataset
from engine.ddp_train_loop import cleanup_ddp, fit_ddp, setup_ddp


# ──────────────────────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────────────────────
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def build_dataloader(cand, hist, lbls, args) -> DataLoader:
    ds = NRMSDataset(cand, hist, lbls)
    sampler = DistributedSampler(ds, shuffle=True)
    workers = args.num_workers or max(os.cpu_count() // int(os.environ["WORLD_SIZE"]), 1)
    return DataLoader(
        ds,
        batch_size=args.batch,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────
def main(args):
    seed_everything(args.seed)

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup_ddp(local_rank, world_size)

    # 1 Data (ideally mem‑mapped to avoid per‑rank duplication)
    # if local_rank == 0:
    #     load_and_prepare_nrms(DATA_DIR, regen=True)
    # torch.distributed.barrier()
    cand, hist, lbls, vocab = load_and_prepare_nrms(DATA_DIR, regen=False)
    dl = build_dataloader(cand, hist, lbls, args)

    # 2 ▸ Model
    from engine.registry import create_model  # late import

    model = create_model(
        "nrms",
        vocab_size=vocab,
        embed_dim=args.embed_dim,
        num_heads=args.heads,
    ).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # 3 ▸ MLflow (rank-0 only)
    if local_rank == 0:
        # ── (a) enable system-metrics sampling every 10 s ────────────────────
        mlflow.system_metrics.enable_system_metrics_logging()
        mlflow.system_metrics.set_system_metrics_sampling_interval(10)

        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        mlflow.pytorch.autolog(  # logs optimiser, LR scheduler, etc.
            log_models=False,  # we checkpoint manually at the end
            disable=False,
        )

        git_sha = None
        try:
            import git

            repo = git.Repo(Path(__file__).resolve().parents[2])
            git_sha = repo.head.commit.hexsha
        except Exception:
            pass

        with mlflow.start_run(
            log_system_metrics=True,  #  ← captures CPU / GPU / I/O
            tags={"git_sha": git_sha} if git_sha else None,
        ) as run:
            # params – every CLI flag
            mlflow.log_params(vars(args))

            # training
            fit_ddp(model, dl, epochs=args.epochs, lr=args.lr, use_amp=args.amp)

            # final model (DDP unwrap)
            mlflow.pytorch.log_model(model.module, artifact_path="model")

            print(f"MLflow run finished: {run.info.run_id}")

    else:
        # non-zero ranks just train
        fit_ddp(model, dl, epochs=args.epochs, lr=args.lr, use_amp=args.amp)

    cleanup_ddp()


if __name__ == "__main__":
    main(parse_args())
