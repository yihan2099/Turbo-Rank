"""Generic, model-agnostic training utilities."""
from __future__ import annotations
import logging, math, torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import mlflow

log = logging.getLogger(__name__)

def _step(model, batch, criterion, optim=None, device="cpu"):
    model.train(optim is not None)
    batch = {k: v.to(device) for k, v in batch.items()}
    # Infer input names
    if "cand" in batch:                          # NRMS
        logits = model(batch["cand"], batch["hist"])
    else:                                        # Two-Tower
        logits = model(batch["user"], batch["item"])
    loss = criterion(logits, batch["label"])
    if optim:
        optim.zero_grad()
        loss.backward()
        optim.step()
    return loss.detach(), torch.sigmoid(logits).detach(), batch["label"].detach()


def fit(model,
        dl: DataLoader,
        epochs: int = 5,
        lr: float = 1e-3,
        criterion = torch.nn.BCEWithLogitsLoss(),
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu"):

    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        losses, preds, lbls = [], [], []
        pbar = tqdm(dl, desc=f"Epoch {ep+1}/{epochs}")
        for batch in pbar:
            loss, p, y = _step(model, batch, criterion, optim, device)
            losses.append(loss.cpu().item())
            preds.extend(p.cpu().numpy())
            lbls.extend(y.cpu().numpy())
            pbar.set_postfix(loss=f"{losses[-1]:.4f}")

        auc  = roc_auc_score(lbls, preds)
        mlflow.log_metric("train_loss", sum(losses)/len(losses), step=ep)
        mlflow.log_metric("train_auc",  auc,                      step=ep)

        log.info("Epoch %d | loss %.4f | AUC %.4f",
                 ep+1, sum(losses)/len(losses), auc)

    # Save final weights – let MLflow manage path
    mlflow.pytorch.log_model(model, artifact_path="model")