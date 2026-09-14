"""
Training & Evaluation Pipeline
Includes Scaffold Splitting, L2 Regularization, Early Stopping, and Model Checkpointing.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.dataset import load_raw_data, scaffold_split, BBBPDataset
from src.model import DeepADMETMLP


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)

    return running_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            running_loss += loss.item() * X_batch.size(0)
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    val_loss = running_loss / len(dataloader.dataset)
    val_auc = roc_auc_score(all_targets, all_preds)
    return val_loss, val_auc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    raw_df = load_raw_data()
    train_df, val_df = scaffold_split(raw_df, train_ratio=0.8)
    print(f"Train set: {len(train_df)} samples | Val set: {len(val_df)} samples")

    train_dataset = BBBPDataset(train_df)
    val_dataset = BBBPDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = DeepADMETMLP(input_dim=2048).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 30
    patience = 5
    patience_counter = 0
    best_val_loss = float("inf")

    print("\nStarting training...")
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val Loss':<12} | {'Val ROC-AUC':<12}")
    print("-" * 50)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        print(
            f"{epoch:<8} | {train_loss:<12.4f} | {val_loss:<12.4f} | {val_auc:<12.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print( f"\nEarly stopping triggered at epoch {epoch}. Saved best checkpoint (Val Loss: {best_val_loss:.4f}).")
                break


if __name__ == "__main__":
    main()