"""Training utilities for StyleSelector model."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    """Get the best available device (MPS → CUDA → CPU).

    Returns:
        torch.device for training
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch.

    Args:
        model: The neural network
        dataloader: Training data loader (returns position, move_idx, target)
        optimizer: Optimizer (e.g., Adam)
        criterion: Loss function (e.g., BCELoss)
        device: Device to train on

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for positions, move_indices, targets in dataloader:
        # Move data to device
        positions = positions.to(device)
        move_indices = move_indices.to(device)
        targets = targets.to(device).float().unsqueeze(1)  # Shape: (batch, 1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(positions, move_indices)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on validation set.

    Args:
        model: The neural network
        dataloader: Validation data loader (returns position, move_idx, target)
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    with torch.no_grad():
        for positions, move_indices, targets in dataloader:
            # Move data to device
            positions = positions.to(device)
            move_indices = move_indices.to(device)
            targets = targets.to(device).float().unsqueeze(1)  # Shape: (batch, 1)

            # Forward pass
            outputs = model(positions, move_indices)

            # Compute loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            num_batches += 1

            # Accuracy: model predicts >0.5 for positive, <0.5 for negative
            predictions = (outputs > 0.5).float()
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    save_dir: Path,
) -> dict:
    """Main training loop with early stopping.

    Args:
        model: The neural network
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Maximum number of epochs
        lr: Learning rate
        device: Device to train on
        save_dir: Directory to save best model

    Returns:
        Training history dict with keys:
        - train_loss: list of training losses per epoch
        - val_loss: list of validation losses per epoch
        - val_accuracy: list of validation accuracies per epoch
        - best_epoch: epoch with best validation loss
    """
    # Setup
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "best_epoch": 0,
    }

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 10  # Early stopping patience

    print(f"Training on {device}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        # Print progress
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Check for best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_epoch"] = epoch
            patience_counter = 0

            # Save best model
            save_path = save_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                save_path,
            )
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    print("-" * 60)
    print(f"Training complete. Best epoch: {history['best_epoch']}")
    print(f"Best val loss: {best_val_loss:.4f}")

    return history
