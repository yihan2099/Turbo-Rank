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
    dl: DataLoader,
    *,
    epochs: int = 5,
    lr: float = 1e-3,
    criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    use_amp: bool = False,
) -> None:
    """Train a DDP-wrapped model."""
    rank = dist.get_rank()
    device = torch.cuda.current_device()

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=use_amp)
    auroc = BinaryAUROC().to(device)

    for ep in range(epochs):
        if isinstance(dl.sampler, DistributedSampler):
            dl.sampler.set_epoch(ep)

        losses = []
        pbar = tqdm(dl, desc=f"R{rank} | Ep {ep + 1}/{epochs}", disable=rank != 0)

        for batch in pbar:
            with autocast(device_type="cuda", enabled=use_amp):
                loss, p, y = _step(model, batch, criterion, device=device)

            # mixed precision backward
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

            losses.append(loss.item())
            auroc.update(p, y)  # ← p already sigmoid

        mean_loss = sum(losses) / len(losses)
        epoch_auc = auroc.compute().item()
        auroc.reset()

        if rank == 0:
            mlflow.log_metric("train_loss", mean_loss, step=ep)
            mlflow.log_metric("train_auc", epoch_auc, step=ep)
            log.info("Epoch %d | loss %.4f | AUC %.4f", ep + 1, mean_loss, epoch_auc)

    if rank == 0:  # final checkpoint
        mlflow.pytorch.log_model(model.module, artifact_path="model")
