# This code is for inference, not for training.

import torch
import torch.nn as nn


class Sign(nn.Module):
    def forward(self, x):
        return torch.where(x >= 0, 1.0, -1.0)


class BNN(nn.Module):
    def __init__(self, input_size=784, hidden_layers=[256, 128], output_size=10):
        super().__init__()

        self.input_block = nn.Sequential(
            nn.BatchNorm1d(input_size, affine=True, eps=2e-5),
            Sign()
        )

        self.hidden_layers = nn.ModuleList()
        in_features = input_size
        for out_features in hidden_layers:
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(in_features, out_features, bias=True),
                nn.BatchNorm1d(out_features, affine=True, eps=2e-5),
                Sign()
            ))
            in_features = out_features

        self.output_layer = nn.Linear(in_features, output_size, bias=True)

    def forward(self, x_scaled):
        y = self.input_block(x_scaled)

        for layer in self.hidden_layers:
            y = layer(y)

        return self.output_layer(y)

    def predict(self, x):
        return self(x / 255.0).argmax(dim=1)
