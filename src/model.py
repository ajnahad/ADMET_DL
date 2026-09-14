"""
Neural Network Architecture Module
Multi-Layer Perceptron (MLP) with Dropout for binary BBB penetration prediction.
"""
import torch
import torch.nn as nn


class DeepADMETMLP(nn.Module):
    def __init__(self, input_dim: int = 2048):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),  
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)