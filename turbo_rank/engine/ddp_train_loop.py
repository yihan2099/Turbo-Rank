"""Distributed training loop with AMP, GradScaler and TorchMetrics."""

from __future__ import annotations

import logging

import mlflow
import torch
from torch.amp import GradScaler, autocast  # ← new import path (PyTorch ≥2.3)
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics.classification import BinaryAUROC
from tqdm import tqdm

from .train_loop import _step  # reuse batch logic

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------


def setup_ddp(rank: int, world_size: int) -> None:
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    torch.cuda.empty_cache()
    dist.destroy_process_group()


# ---------------------------------------------------------------------------


def fit_ddp(
    model: torch.nn.Module,
    dl_train: DataLoader,
    dl_val: DataLoader | None = None,  # ← NEW
    *,
    epochs: int = 5,
    lr: float = 1e-3,
    criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    use_amp: bool = False,
) -> float:  # returns final val-AUC
    rank = dist.get_rank()
    device = torch.cuda.current_device()

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=use_amp)
    auroc = BinaryAUROC().to(device)

    global_step = 0

    best_auc = 0.0
    for ep in range(epochs):
        if isinstance(dl_train.sampler, DistributedSampler):
            dl_train.sampler.set_epoch(ep)

        losses = []
        pbar = tqdm(dl_train, desc=f"R{rank}|Ep {ep + 1}/{epochs}", disable=rank != 0)

        for batch in pbar:
            with autocast(device_type="cuda", enabled=use_amp):
                loss, p, y = _step(model, batch, criterion, device=device)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

            # ───── per-batch logging ──────────────────────────────────────
            if rank == 0:  # only one process logs
                mlflow.log_metric("batch_loss", loss.item(), step=global_step)
            global_step += 1  # increment every batch
            # -----------------------------------------------------------------

            losses.append(loss.item())
            auroc.update(p, y)

        train_loss = sum(losses) / len(losses)
        train_auc = auroc.compute().item()
        auroc.reset()

        # -------- validation ---------------------------------------------
        if dl_val is not None:
            val_loss, val_auc = evaluate_ddp(
                model, dl_val, device=device, criterion=criterion, use_amp=use_amp
            )
            best_auc = max(best_auc, val_auc)
        else:
            val_loss = val_auc = None

        if rank == 0:
            mlflow.log_metric("train_loss", train_loss, step=ep)
            mlflow.log_metric("train_auc", train_auc, step=ep)
            if dl_val is not None:
                mlflow.log_metric("val_loss", val_loss, step=ep)
                mlflow.log_metric("val_auc", val_auc, step=ep)

    # last checkpoint
    if rank == 0:
        mlflow.pytorch.log_model(model.module, artifact_path="model")
    return best_auc


@torch.no_grad()
def evaluate_ddp(model, dl, *, device, criterion, use_amp: bool):
    # rank   = dist.get_rank()
    auroc = BinaryAUROC().to(device)
    losses = []

    for batch in dl:
        with autocast(device_type="cuda", enabled=use_amp):
            loss, p, y = _step(model, batch, criterion, device=device)
        losses.append(loss.item())
        auroc.update(p, y)

    return sum(losses) / len(losses), auroc.compute().item()
