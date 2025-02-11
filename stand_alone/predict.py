import argparse

import torch
import torch.nn as nn
import numpy as np
from PIL import Image


class Sign(nn.Module):
    def forward(self, x):
        return torch.where(x >= 0, 1.0, -1.0)


class BNN(nn.Module):
    def __init__(self, input_size=784, hidden_layers=[256, 128], output_size=10):
        super().__init__()

        self.input_block = nn.Sequential(
            nn.BatchNorm1d(input_size, affine=True, eps=2e-5), Sign()
        )

        self.hidden_layers = nn.ModuleList()
        in_features = input_size
        for out_features in hidden_layers:
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, out_features, bias=True),
                    nn.BatchNorm1d(out_features, affine=True, eps=2e-5),
                    Sign(),
                )
            )
            in_features = out_features

        self.output_layer = nn.Linear(in_features, output_size, bias=True)

    def forward(self, x_scaled):
        y = self.input_block(x_scaled)

        for layer in self.hidden_layers:
            y = layer(y)

        return self.output_layer(y)

    def predict(self, x):
        return self(x / 255.0).argmax(dim=1)


parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="PyTorch model (*.pt)")
parser.add_argument("image", type=str, help="input image (*.png)")
parser.add_argument(
    "--column-major",
    action="store_true",
    help="feature vector is in column major order",
)
args = parser.parse_args()

model = BNN(hidden_layers=[200, 100, 100, 100])
state_dict = torch.load(args.model, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

x = np.array(Image.open(args.image))
if args.column_major:
    x = x.T
x = x.reshape(-1)

with torch.inference_mode():
    logits = model(torch.tensor(x[None, :] / 255.0, dtype=torch.float32))
    prob = nn.functional.softmax(logits, dim=1)
    print(f"  logits: {logits[0].numpy().tolist()}")
    print(f"  probability: {prob[0].numpy().tolist()}")
    print(f"  predicted class: {np.argmax(logits[0].numpy())}")
