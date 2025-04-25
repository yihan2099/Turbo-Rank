# turbo_rank/cli/train_nrms_ddp.py
# ------------------------------------------------------------
"""
CLI wrapper for multi-GPU NRMS training.

Launch via:
    torchrun --standalone --nproc_per_node=4 cli/train_nrms_ddp.py \
        --epochs 3 --batch 1024
"""
from __future__ import annotations
import os, argparse, torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from config.paths       import DATA_DIR, MLFLOW_EXPERIMENT
from data.preprocess    import load_and_prepare_nrms
from datasets.nrms      import NRMSDataset
from engine.registry    import create_model
from engine.ddp_train_loop import _setup_ddp, _cleanup_ddp, fit_ddp

# ------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--heads",     type=int, default=8)
    ap.add_argument("--epochs",    type=int, default=3)
    ap.add_argument("--batch",     type=int, default=64)
    ap.add_argument("--lr",        type=float, default=1e-3)
    return ap.parse_args()

# ------------------------------------------------------------------
def main():
    rank       = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    _setup_ddp(rank, world_size)

    # 1 Data
    cand, hist, lbls, vocab = load_and_prepare_nrms(DATA_DIR)
    ds = NRMSDataset(cand, hist, lbls)
    sampler = DistributedSampler(ds, shuffle=True, drop_last=False)
    dl = DataLoader(ds,
                    batch_size=args.batch,
                    sampler=sampler,
                    num_workers=4,
                    pin_memory=True)

    # 2 Model
    model = create_model("nrms",
                         vocab_size=vocab,
                         embed_dim=args.embed_dim,
                         num_heads=args.heads)
    model = DDP(model.cuda(rank), device_ids=[rank], output_device=rank)

    # 3 MLflow (rank-0 only touches the store)
    if rank == 0:
        import mlflow
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        mlflow.start_run()

    fit_ddp(model,
            dl,
            epochs=args.epochs,
            lr=args.lr)

    if rank == 0:
        mlflow.end_run()

    _cleanup_ddp()

# ------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    main()