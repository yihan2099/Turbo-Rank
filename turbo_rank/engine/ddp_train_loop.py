# turbo_rank/engine/ddp_train_loop.py
# ------------------------------------------------------------
"""Distributed (DDP) variant of the generic training loop."""

from __future__ import annotations

import logging

import mlflow
from sklearn.metrics import roc_auc_score
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from .train_loop import _step  # ← reuse your existing logic

log = logging.getLogger(__name__)


# -----------------------------------------------------------------
def _setup_ddp(rank: int, world_size: int):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _cleanup_ddp():
    dist.destroy_process_group()


# -----------------------------------------------------------------
def fit_ddp(
    model: torch.nn.Module,
    dl: DataLoader,
    epochs: int = 5,
    lr: float = 1e-3,
    criterion=torch.nn.BCEWithLogitsLoss(),
):
    """Same contract as *fit()* but executed inside a DDP process.

    Assumes:
      • torch.cuda.set_device(rank) already called
      • model already wrapped with DistributedDataParallel
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.cuda.current_device()
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # -----------------------------------------------------------------
    for ep in range(epochs):
        # DistributedSampler must reshuffle each epoch
        if isinstance(dl.sampler, DistributedSampler):
            dl.sampler.set_epoch(ep)

        losses, preds, lbls = [], [], []
        pbar = tqdm(dl, desc=f"Rank{rank} | Epoch {ep + 1}/{epochs}", disable=rank != 0)

        for batch in pbar:
            loss, p, y = _step(model, batch, criterion, optim, device)
            losses.append(loss.item())
            preds.append(p)
            lbls.append(y)

        # --- gather tensors from all ranks ---------------------------
        with torch.no_grad():
            p_cat = torch.cat(preds)
            y_cat = torch.cat(lbls)
            gathered_p = [torch.zeros_like(p_cat) for _ in range(world_size)]
            gathered_y = [torch.zeros_like(y_cat) for _ in range(world_size)]
            dist.all_gather(gathered_p, p_cat)
            dist.all_gather(gathered_y, y_cat)

            if rank == 0:
                y_true = torch.cat(gathered_y).cpu().numpy()
                y_pred = torch.cat(gathered_p).cpu().numpy()
                auc = roc_auc_score(y_true, y_pred)
                mean_loss = sum(losses) / len(losses)

                mlflow.log_metric("train_loss", mean_loss, step=ep)
                mlflow.log_metric("train_auc", auc, step=ep)
                log.info("Epoch %d | loss %.4f | AUC %.4f", ep + 1, mean_loss, auc)

    # -----------------------------------------------------------------
    # Check-point once
    if rank == 0:
        mlflow.pytorch.log_model(model.module, artifact_path="model")
