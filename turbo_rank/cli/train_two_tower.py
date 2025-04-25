from argparse import ArgumentParser
import mlflow
from config.paths import DATA_DIR, MLFLOW_EXPERIMENT
from data.preprocess import load_and_prepare_two_tower
from data.encoders   import dump_encoder_tmp
from datasets.two_tower import MindInteractionDataset
from engine.registry import create_model
from engine.train_loop import fit
from torch.utils.data import DataLoader

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--emb_dim",  type=int, default=64)
    ap.add_argument("--epochs",   type=int, default=5)
    ap.add_argument("--batch",    type=int, default=1024)
    args = ap.parse_args()

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run():
        samples, u_enc, i_enc = load_and_prepare_two_tower(DATA_DIR)
        ds = MindInteractionDataset(samples)
        dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=4, pin_memory=True)

        model = create_model("two_tower",
                             num_users=len(u_enc.classes_),
                             num_items=len(i_enc.classes_),
                             emb_dim=args.emb_dim)

        fit(model, dl, epochs=args.epochs)

        # dump encoders so inference scripts can recover them
        mlflow.log_artifact(dump_encoder_tmp(u_enc), artifact_path="artifacts")
        mlflow.log_artifact(dump_encoder_tmp(i_enc), artifact_path="artifacts")