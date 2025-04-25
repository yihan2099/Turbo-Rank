"""Generic, model-agnostic training utilities."""

from __future__ import annotations

import logging
from typing import Tuple

import torch

log = logging.getLogger(__name__)


def _step(
    model,
    batch: dict[str, torch.Tensor],
    criterion,
    *,
    device: str | torch.device = "cuda",
    optim=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass + *optional* parameter update.

    Returns
    -------
    loss : **NOT detached** so the caller can backward() or scale().backward().
    preds : sigmoid probabilities **detached** (B,)
    labels: ground-truth labels **detached** (B,)
    """
    # train() only if optim is supplied (keeps eval behaviour elsewhere)
    model.train(optim is not None)

    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

    if "cand" in batch:  # NRMS
        logits = model(batch["cand"], batch["hist"])
    else:  # Two-Tower
        logits = model(batch["user"], batch["item"])

    loss = criterion(logits, batch["label"])

    if optim is not None:
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

    # ────────────────────────────────────────────────────────────────────
    preds = torch.sigmoid(logits.detach())
    labels = batch["label"].detach()
    return loss, preds, labels
