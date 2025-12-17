"""StyleSelector neural network for predicting player move preferences."""

import torch
import torch.nn as nn


class StyleSelector(nn.Module):
    """Neural network that predicts probability a player would make a given move.

    Takes a chess position (12x8x8 tensor) and a candidate move index,
    outputs probability that the player would choose this move.
    """

    def __init__(self):
        super().__init__()

        # Convolutional layers for position encoding
        # 12 input channels (one per piece type)
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Move embedding: 4096 possible moves -> 64-dim embedding
        self.move_embedding = nn.Embedding(4096, 64)

        # Fully connected layers
        # Input: flattened conv (128 * 8 * 8 = 8192) + move embedding (64) = 8256
        self.fc1 = nn.Linear(8192 + 64, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, position: torch.Tensor, move_idx: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            position: Board position tensor of shape (batch, 12, 8, 8)
            move_idx: Move indices of shape (batch,) with values in [0, 4095]

        Returns:
            Probability tensor of shape (batch, 1)
        """
        # Convolutional layers
        x = self.relu(self.conv1(position))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Flatten conv output: (batch, 128, 8, 8) -> (batch, 8192)
        x = x.view(x.size(0), -1)

        # Get move embedding: (batch,) -> (batch, 64)
        move_emb = self.move_embedding(move_idx)

        # Concatenate position features and move embedding
        x = torch.cat([x, move_emb], dim=1)

        # Fully connected layers with dropout
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))

        return x


def get_model_size(model: nn.Module) -> int:
    """Get total number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Total parameter count
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
