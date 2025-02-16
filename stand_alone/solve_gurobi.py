import math
from typing import Literal

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import gurobipy as gp
from gurobipy import GRB


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


def create_model(model, x_orig, y_true, norm: Literal["0", "1", "2", "inf"]):
    # Create optimization model
    m = gp.Model("adversarial_bnn")

    # Input dimension
    input_dim = 784

    # Extract batch norm parameters from input block
    bn0 = model.input_block[0]
    gamma0 = bn0.weight.data.numpy()
    beta0 = bn0.bias.data.numpy()
    mu0 = bn0.running_mean.data.numpy()
    var0 = bn0.running_var.data.numpy()
    eps0 = bn0.eps
    sigma0 = np.sqrt(var0 + eps0)

    # Input image after binalization
    input_bin = m.addVars(input_dim, vtype=GRB.BINARY, name="input_bin")
    orig_bin = gamma0 * (x_orig / np.float32(255.0) - mu0) / sigma0 + beta0 >= 0

    mod = []
    for j in range(len(orig_bin)):
        C_frac = (-beta0[j] * sigma0[j] / gamma0[j] + mu0[j]) * np.float32(255.0)

        if gamma0[j] >= 0:
            # x ≥ ⌈255 (- βσ/γ + μ)⌉ = C
            C = int(math.ceil(C_frac))
            if orig_bin[j] != (x_orig[j] >= C):
                raise RuntimeError("numerically unstable")
            if x_orig[j] < C:
                if C <= 255:
                    min_diff = C - int(x_orig[j])
                else:
                    input_bin[j].UB = 0
                    min_diff = None
                mod.append((True, min_diff))
            else:
                if C - 1 >= 0:
                    min_diff = (C - 1) - int(x_orig[j])
                else:
                    input_bin[j].LB = 1
                    min_diff = None
                mod.append((False, min_diff))
        else:
            # x ≤ ⌊255 (- βσ/γ + μ)⌋ = C
            C = int(math.floor(C_frac))
            if orig_bin[j] != (x_orig[j] <= C):
                raise RuntimeError("numerically unstable")
            if x_orig[j] > C:
                if C >= 0:
                    min_diff = C - int(x_orig[j])
                else:
                    input_bin[j].LB = 1
                    min_diff = None
                mod.append((True, min_diff))
            else:
                if C + 1 <= 255:
                    min_diff = (C + 1) - int(x_orig[j])
                else:
                    input_bin[j].UB = 0
                    min_diff = None
                mod.append((False, min_diff))

    xs = input_bin

    # Intermediate layers
    for layer_idx, layer in enumerate(model.hidden_layers):
        linear = layer[0]
        bn = layer[1]

        weights = linear.weight.data.sign().numpy()
        bias = linear.bias.data.numpy()
        n_out = weights.shape[0]

        gamma = bn.weight.data.numpy()
        beta = bn.bias.data.numpy()
        mu = bn.running_mean.data.numpy()
        var = bn.running_var.data.numpy()
        eps = bn.eps
        sigma = np.sqrt(var + eps)

        y = m.addVars(n_out, vtype=GRB.BINARY, name=f"block{layer_idx}")

        for i in range(n_out):
            # γ((Σ_j row[j]*(2*xs[j]-1) + b) - μ)/σ + β ≥ 0
            row = weights[i]
            lhs = gp.quicksum(row[j] * x for (j, x) in xs.items())
            C_frac = (-beta[i] * sigma[i] / gamma[i] + mu[i] - bias[i] + sum(row)) / 2
            if gamma[i] >= 0:
                # Σ_j row[j]*xs[j] ≥ ⌈(-βσ/γ + μ - b + Σ_j row[j]) / 2⌉ = C
                C = int(math.ceil(C_frac))
                m.addConstr((y[i] == 1) >> (lhs >= C))
                m.addConstr((y[i] == 0) >> (lhs <= C - 1))
            else:
                # Σ_j row[j]*xs[j] ≤ ⌊(-βσ/γ + μ - b + Σ_j row[j]) / 2⌋ = C
                C = int(math.floor(C_frac))
                m.addConstr((y[i] == 1) >> (lhs <= C))
                m.addConstr((y[i] == 0) >> (lhs >= C + 1))

        xs = y

    # Output layer
    n_out = 10
    output = m.addVars(n_out, vtype=GRB.BINARY, name="output")
    m.addConstr(gp.quicksum(output[i] for i in range(n_out)) == 1)
    weights = model.output_layer.weight.data.sign().numpy()
    bias = model.output_layer.bias.data.sign().numpy()
    logits = [(2 * weights[i, :], -sum(weights[i, :]) + bias[i]) for i in range(n_out)]
    for i in range(n_out):
        for j in range(n_out):
            if i == j:
                continue
            # logits[i] ≥ logits[j]
            lhs = [
                (w, xs[k]) for k, w in enumerate(logits[i][0] - logits[j][0]) if w != 0
            ]
            rhs = int(math.ceil(logits[j][1] - logits[i][1]))
            assert all(c % 4 == 0 for c, _ in lhs)
            lhs = [(c // 4, v) for c, v in lhs]
            rhs = (rhs + 3) // 4  # Note that // is floor division
            m.addConstr((output[i] == 1) >> (gp.quicksum(c * v for c, v in lhs) >= rhs))

    # Misclassification constraint
    output[y_true].UB = 0

    # Objective function
    if norm == "inf":
        top = m.addVar(name="top")
        for i, (target, w) in enumerate(mod):
            if w is None:
                continue
            e = input_bin[i] if target else 1 - input_bin[i]
            m.addConstr(abs(w) * e <= top)
        m.setObjective(top, GRB.MINIMIZE)
    else:
        obj = []
        for i, (target, w) in enumerate(mod):
            if w is None:
                continue
            e = input_bin[i] if target else 1 - input_bin[i]
            match norm:
                case "0":
                    obj.append(e)
                case "1":
                    obj.append(abs(w) * e)
                case "2":
                    obj.append(w**2 * e)
                case _:
                    raise RuntimeError(f"unknown norm: {norm}")
        m.setObjective(gp.quicksum(obj), GRB.MINIMIZE)

    m.update()
    return m, input_bin, output, mod


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="model file (*.pt)")
    parser.add_argument("input", type=str, help="input image filename (*.png)")
    parser.add_argument("label", type=int, help="label")
    parser.add_argument(
        "--column-major",
        action="store_true",
        help="feature vector is in column major order",
    )
    parser.add_argument(
        "--norm",
        type=str,
        choices=["0", "1", "2", "inf"],
        default="inf",
        help="norm to minimize",
    )
    parser.add_argument(
        "-o", "--output", type=str, help="output image filename (*.png)"
    )
    parser.add_argument(
        "--initial-solution", type=str, help="initial solution filename (*.sol)"
    )
    parser.add_argument(
        "--output-solution", type=str, help="output solution filename (*.sol)"
    )
    parser.add_argument(
        "--output-problem", type=str, help="output problem filename (*.mps or *.lp)"
    )
    args = parser.parse_args()

    # Load model
    model = BNN(hidden_layers=[200, 100, 100, 100])
    state_dict = torch.load(args.model, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Load input image
    x_orig = np.array(Image.open(args.input))
    if args.column_major:
        x_orig = x_orig.T
    x_orig = x_orig.reshape(-1)

    # Load true label
    y_true = args.label

    # Create MILP model
    m, input_bin, output, mod = create_model(model, x_orig, y_true, norm=args.norm)
    if args.output_problem:
        m.write(args.output_problem)

    # Solve MILP model
    if args.initial_solution is not None:
        m.read(args.initial_solution)
    # m.Params.NoRelHeurTime = 600
    m.optimize()

    # Extract solution
    if m.SolCount > 0:
        if args.output_solution:
            m.write(args.output_solution)

        x_bin = np.array([input_bin[i].x >= 0.5 for i in range(784)], dtype=bool)
        diff = np.array(
            [
                min_diff if x_bin[i] == target else 0
                for i, (target, min_diff) in enumerate(mod)
            ],
            dtype=np.int32,
        )
        x_perturbed = (x_orig + diff).astype(np.uint8)

        print("Perturbation:")
        diff = diff.reshape(28, 28)
        if args.column_major:
            diff = diff.T
        print(diff)
        for norm in [0, 1, 2, np.inf]:
            z = np.linalg.norm(diff.reshape(-1), ord=norm)
            print(f"{norm}-norm: {z}")

        # Save perturbed image
        if args.output:
            x_perturbed = x_perturbed.reshape(28, 28)
            if args.column_major:
                x_perturbed = x_perturbed.T
            Image.fromarray(x_perturbed.astype(np.uint8)).save(args.output)


if __name__ == "__main__":
    main()
