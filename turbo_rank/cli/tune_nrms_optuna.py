"""
Hyper-parameter sweep for NRMS using Optuna + MLflow
Run with:
    PYTHONPATH=. python cli/tune_nrms_optuna.py --trials 50 --gpus 4
    optuna-dashboard sqlite:///db.sqlite3 --port 4000    # or point to your RDB backend
    mlflow ui
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path
import subprocess

import mlflow
import optuna

from turbo_rank.config.paths import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI

REPO_ROOT = Path(__file__).resolve().parents[1]

SWEEP_NAME = "NRMS_optuna_sweep"


# ----------------------------------------------------------------------
def build_cli_cmd(trial, n_gpus):
    """Return the shell command that launches one NRMS training."""
    # ---- sample hyper-params ------------------------------------------------
    embed_dim = trial.suggest_categorical("embed_dim", [128, 256, 384])
    heads = trial.suggest_categorical("heads", [4, 8, 12])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch = trial.suggest_categorical("batch", [64, 128, 256])

    # bail out early if incompatible
    if embed_dim % heads != 0:  # ⇦ key line
        raise optuna.TrialPruned(f"{embed_dim=} not divisible by {heads=}")

    # ---- build torchrun command --------------------------------------------
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(n_gpus)))
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"

    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={n_gpus}",
        "cli/train_nrms_ddp.py",
        "--epochs",
        "4",
        "--amp",
        "--batch",
        str(batch),
        "--embed_dim",
        str(embed_dim),
        "--heads",
        str(heads),
        "--lr",
        f"{lr:.5g}",
        "--num_workers",
        str(max(mp.cpu_count() // n_gpus, 1)),
    ]
    return cmd, env


def objective(trial, n_gpus):
    # -------- child run (one per trial) --------------------
    with mlflow.start_run(
        run_name=f"trial_{trial.number}",
        nested=True,  # ← child of the sweep run
        tags=trial.params,  # record hyper-params immediately
    ) as active:
        run_id = active.info.run_id
        cmd, env = build_cli_cmd(trial, n_gpus)
        env["MLFLOW_RUN_ID"] = run_id
        completed = subprocess.run(
            cmd, env=env, cwd=str(REPO_ROOT), capture_output=True, text=True
        )
        if completed.returncode != 0:
            raise optuna.TrialPruned(completed.stderr)

        # read metric the worker just logged
        client = mlflow.MlflowClient()
        auc = client.get_metric_history(run_id, "val_auc")[-1].value
        return auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--gpus", type=int, default=1)
    args = ap.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=SWEEP_NAME) as sweep_run:
        study = optuna.create_study(
            storage="sqlite:///db.sqlite3",
            direction="maximize",
            study_name=SWEEP_NAME,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )
        study.optimize(lambda t: objective(t, args.gpus), n_trials=args.trials)

        # ─── log the best trial’s params & value ─────────────────────────────
        best = study.best_trial
        # logs each hyperparam as an MLflow parameter
        mlflow.log_params(best.params)
        # log the best objective (e.g. validation AUC) as a metric
        mlflow.log_metric("best_val_auc", best.value)

        # optional: tag the best-trial-number for easy filtering later
        mlflow.set_tag("optuna.best_trial", best.number)

    # console summary
    print("Best trial #", best.number)
    print("  params:", best.params)
    print("  value :", best.value)
    print("MLflow sweep run ID:", sweep_run.info.run_id)


if __name__ == "__main__":
    main()
