import logging
import argparse
from pathlib import Path
import mlflow

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import joblib # For saving encoders

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Use Path for better path handling
PROJECT_ROOT = Path(__file__).resolve().parents[2] # Assumes script is in turbo_rank/models/
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "baseline"
MODEL_DIR.mkdir(parents=True, exist_ok=True) # Ensure model directory exists

# --- Data Loading and Preprocessing ---

def load_data(data_dir: Path, split: str = "train") -> pd.DataFrame:
    """Loads behaviors data from parquet files."""
    behaviors_path = data_dir / split / "behaviors.parquet"
    logging.info(f"Loading behaviors data from: {behaviors_path}")
    if not behaviors_path.exists():
        raise FileNotFoundError(f"Processed behaviors data not found at {behaviors_path}. Run preprocessing first.")
    # news_path = data_dir / split / "news.parquet" # Load news if needed later
    # logging.info(f"Loading news data from: {news_path}")
    # news_df = pd.read_parquet(news_path)
    behaviors_df = pd.read_parquet(behaviors_path)
    return behaviors_df #, news_df

def parse_impressions(impressions: list[str]) -> list[tuple[str, int]]:
    """Parses a list of impression strings into (item_id, label) tuples."""
    parsed = []
    if impressions is None: # Handle cases where impressions might be null
        return parsed
    for impression in impressions:
        try:
            item_id, label_str = impression.split('-')
            label = int(label_str)
            parsed.append((item_id, label))
        except ValueError:
            logging.warning(f"Could not parse impression: {impression}. Skipping.")
            continue
    return parsed

def create_training_samples(behaviors_df: pd.DataFrame) -> pd.DataFrame:
    """Transforms behaviors data into (user_id, item_id, label) samples."""
    logging.info("Creating training samples from behaviors data...")
    # Use explode for potentially better performance than iterrows on large data
    behaviors_df['parsed_impressions'] = behaviors_df['impressions'].apply(parse_impressions)
    samples_df = behaviors_df.explode('parsed_impressions')
    samples_df = samples_df[samples_df['parsed_impressions'].notna()] # Drop rows where parsing failed or impressions were empty

    samples_df[['item_id', 'label']] = pd.DataFrame(samples_df['parsed_impressions'].tolist(), index=samples_df.index)
    samples_df = samples_df[['user_id', 'item_id', 'label']].copy() # Select and copy relevant columns
    logging.info(f"Generated {len(samples_df)} training samples.")
    return samples_df

def encode_ids(df: pd.DataFrame, user_encoder: LabelEncoder = None, item_encoder: LabelEncoder = None) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """Encodes user and item IDs using LabelEncoder. Fits if encoders are not provided."""
    logging.info("Encoding user and item IDs...")
    if user_encoder is None:
        user_encoder = LabelEncoder()
        df["user_id_enc"] = user_encoder.fit_transform(df["user_id"])
        logging.info(f"Fitted user encoder with {len(user_encoder.classes_)} unique users.")
    else:
        # Handle unseen users during inference/validation if necessary
        df["user_id_enc"] = user_encoder.transform(df["user_id"])

    if item_encoder is None:
        item_encoder = LabelEncoder()
        df["item_id_enc"] = item_encoder.fit_transform(df["item_id"])
        logging.info(f"Fitted item encoder with {len(item_encoder.classes_)} unique items.")
    else:
         # Handle unseen items during inference/validation if necessary
        df["item_id_enc"] = item_encoder.transform(df["item_id"])

    return df, user_encoder, item_encoder

# --- PyTorch Dataset and Model ---

class MindDataset(Dataset):
    """PyTorch Dataset for MIND data."""
    def __init__(self, df: pd.DataFrame):
        self.user_ids = df["user_id_enc"].values
        self.item_ids = df["item_id_enc"].values
        self.labels = df["label"].values

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "user": torch.tensor(self.user_ids[idx], dtype=torch.long),
            "item": torch.tensor(self.item_ids[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }

class TwoTowerModel(nn.Module):
    """Simple Two-Tower recommendation model."""
    def __init__(self, num_users: int, num_items: int, emb_dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim, sparse=False) # sparse=True can sometimes be faster but check compatibility
        self.item_emb = nn.Embedding(num_items, emb_dim, sparse=False)
        # Consider adding more layers or dropout
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2, 128),
            nn.ReLU(),
            # nn.Dropout(0.2), # Optional dropout
            nn.Linear(128, 1),
            # No Sigmoid here if using BCEWithLogitsLoss
        )

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        user_vec = self.user_emb(user)
        item_vec = self.item_emb(item)
        x = torch.cat([user_vec, item_vec], dim=-1)
        # Return logits, loss function will handle sigmoid
        return self.fc(x).squeeze(-1)

# --- Training Loop ---

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    # Wrap dataloader with tqdm for a progress bar
    pbar = tqdm(dataloader, desc="Training Epoch")
    for batch in pbar:
        users = batch["user"].to(device)
        items = batch["item"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(users, items)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # Apply sigmoid here for AUC calculation, as model outputs logits
        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar description
        pbar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_preds)
    return avg_loss, auc

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    # val_loader: DataLoader, # Add validation loader if available
    epochs: int = 5,
    lr: float = 1e-3,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> nn.Module:
    """Full training loop."""
    logging.info(f"Starting training on device: {device}")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Use BCEWithLogitsLoss as it's more numerically stable and expects raw logits
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, device)
        # --- Optional: Validation Step ---
        # model.eval()
        # val_loss, val_auc = evaluate(model, val_loader, criterion, device) # Implement evaluate function
        # logging.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
        logging.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")

    logging.info("Training finished.")
    return model

# --- Main Execution ---

def main(args):
    logging.info("--- Starting Baseline Model Training ---")
    
    experiment_name = "baseline_two_tower_mind"

    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(
            name=experiment_name,
        )
    
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Log params
        mlflow.log_param("embedding_dim", args.emb_dim)
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("epochs", args.epochs)

        # 1. Load Data
        behaviors_df = load_data(DATA_DIR, split="train")

        # 2. Create Samples
        sample_df = create_training_samples(behaviors_df)

        # 3. Encode IDs
        sample_df, user_encoder, item_encoder = encode_ids(sample_df)

        # 4. Create Dataset and DataLoader
        dataset = MindDataset(sample_df)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        # 5. Initialize Model
        num_users = len(user_encoder.classes_)
        num_items = len(item_encoder.classes_)
        model = TwoTowerModel(num_users=num_users, num_items=num_items, emb_dim=args.emb_dim)

        # 6. Train Model
        trained_model = train_model(model, dataloader, epochs=args.epochs, lr=args.lr)

        # 7. Log metrics (last epoch only here)
        train_loss, train_auc = train_epoch(model, dataloader, optimizer=torch.optim.Adam(model.parameters()), criterion=nn.BCEWithLogitsLoss(), device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        mlflow.log_metric("final_train_loss", train_loss)
        mlflow.log_metric("final_train_auc", train_auc)

        # 8. Log model and encoders
        mlflow.pytorch.log_model(trained_model, artifact_path="model")

        model_save_path = MODEL_DIR / "two_tower_model.pth"
        user_encoder_path = MODEL_DIR / "user_encoder.joblib"
        item_encoder_path = MODEL_DIR / "item_encoder.joblib"

        torch.save(trained_model.state_dict(), model_save_path)
        joblib.dump(user_encoder, user_encoder_path)
        joblib.dump(item_encoder, item_encoder_path)

        mlflow.log_artifact(model_save_path)
        mlflow.log_artifact(user_encoder_path)
        mlflow.log_artifact(item_encoder_path)

        logging.info("--- Baseline Model Training Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline Two-Tower Model")
    parser.add_argument("--emb_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    # Add arguments for data paths if needed, but using relative paths for now

    args = parser.parse_args()
    main(args)