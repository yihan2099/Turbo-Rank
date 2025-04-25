"""
Hyper-parameter sweep for NRMS using Optuna + MLflow
Run with:
    python -m cli.tune_nrms_optuna --trials 50 --gpus 4
    optuna-dashboard sqlite:///{}.db  # or point to your RDB backend
    mlflow ui
"""

from __future__ import annotations
import argparse, os, multiprocessing as mp, subprocess, json, uuid, sys, tempfile
from pathlib import Path

import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback

REPO_ROOT = Path(__file__).resolve().parents[2]

# ----------------------------------------------------------------------
def build_cli_cmd(trial, n_gpus):
    """Return the shell command that launches one NRMS training."""
    # ---- sample hyper-params ------------------------------------------------
    embed_dim   = trial.suggest_categorical("embed_dim", [128, 256, 384])
    heads       = trial.suggest_categorical("heads",     [4, 8, 12])
    lr          = trial.suggest_loguniform("lr", 1e-4, 3e-3)
    batch       = trial.suggest_categorical("batch", [64, 128, 256])

    # ---- build torchrun command --------------------------------------------
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(n_gpus)))

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", f"--nproc_per_node={n_gpus}",
        str(REPO_ROOT / "cli" / "train_nrms_ddp.py"),
        "--epochs", "4",
        "--amp",
        "--batch", str(batch),
        "--embed_dim", str(embed_dim),
        "--heads", str(heads),
        "--lr", f"{lr:.5g}",
        "--num_workers", str(max(mp.cpu_count() // n_gpus, 1)),
    ]
    return cmd, env


def objective(trial, n_gpus):
    cmd, env = build_cli_cmd(trial, n_gpus)

    # run; MLflow autolog inside the script logs the val-AUC metric
    with tempfile.TemporaryDirectory() as tmp:
        run_id_file = Path(tmp) / "run_id.txt"
        env["MLFLOW_RUN_ID_FILE"] = str(run_id_file)  # custom: store run-id

        completed = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if completed.returncode != 0:
            # mark the trial as failed
            raise optuna.TrialPruned(f"script failed:\n{completed.stderr}")

        # load val-AUC from MLflow
        run_id = run_id_file.read_text().strip()
        client = mlflow.tracking.MlflowClient()
        auc    = client.get_metric_history(run_id, "val_auc")[-1].value
        trial.set_user_attr("mlflow_run_id", run_id)
        return auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--gpus",   type=int, default=1)
    args = ap.parse_args()

    study = optuna.create_study(
        direction="maximize",
        pruner   = optuna.pruners.MedianPruner(n_startup_trials=5),
        study_name=f"NRMS-{uuid.uuid4().hex[:6]}",
    )
    mlflc = MLflowCallback(
        tracking_uri = mlflow.get_tracking_uri(),
        metric_name  = "val_auc",
        nest_trials  = True,
    )
    study.optimize(
        lambda t: objective(t, args.gpus),
        n_trials=args.trials,
        callbacks=[mlflc],
    )

    print("Best trial:")
    print(study.best_trial.params)
    print("MLflow run:", study.best_trial.user_attrs["mlflow_run_id"])


if __name__ == "__main__":
    main()