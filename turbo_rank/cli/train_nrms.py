from argparse import ArgumentParser
import mlflow, torch.nn as nn
from config.paths import DATA_DIR, MLFLOW_EXPERIMENT
from data.preprocess import load_and_prepare_nrms
from datasets.nrms import NRMSDataset
from engine.registry import create_model
from engine.train_loop import fit
from torch.utils.data import DataLoader

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--embed_dim", type=int, default=288)
    ap.add_argument("--heads",     type=int, default=8)
    ap.add_argument("--epochs",    type=int, default=3)
    ap.add_argument("--batch",     type=int, default=128)
    args = ap.parse_args()

    cand, hist, lbls, vocab = load_and_prepare_nrms(DATA_DIR)
    ds = NRMSDataset(cand, hist, lbls)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=4, pin_memory=True)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run():
        model = create_model("nrms",
                             vocab_size=vocab,
                             embed_dim=args.embed_dim,
                             num_heads=args.heads)

        fit(model, dl, epochs=args.epochs,
            criterion=nn.BCEWithLogitsLoss())