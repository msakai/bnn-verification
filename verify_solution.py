import argparse
import math
from pathlib import Path
import re
from typing import Optional, Sequence, Union

import chainer
import chainer.functions as F
import numpy as np

import bnn
import datasets
import visualize


def read_binary_solution_maxsat(fname: Union[Path, str]) -> Optional[np.ndarray]:
    sol = np.zeros(28*28, dtype=bool)
    with open(fname) as f:
        for line in f:
            # hack for parsing the output of CASHWMaxSAT solvers from MaxSAT Evaluation 2024
            if (idx := line.find("\t")) >= 0:
                line = line[idx+1:]
            if line.startswith('v '):
                line = line[2:].strip()
                if re.fullmatch(r"[01]+", line):
                    for i in range(28*28):
                        sol[i] = (line[i] == "1")
                    break
                else:
                    for s in line.split():
                        l = int(s)
                        if abs(l) <= 28*28:
                            sol[abs(l) - 1] = (l > 0)
    return sol


def read_binary_solution_gurobi(fname: Union[Path, str]) -> Optional[np.ndarray]:
    sol = np.zeros(28*28, dtype=bool)
    p = re.compile(r"input_bin\((\d+)\)\s+(\d+)")
    with open(fname) as f:
        for line in f:
            if m := re.fullmatch(p, line.strip()):
                sol[int(m.group(1))] = int(m.group(2))
    return sol


def decode_binary_solution(model, orig_image: np.ndarray, sol: np.ndarray) -> np.ndarray:
    perturbated_image = orig_image.copy()

    input_bn = model.input_bn
    mu = input_bn.avg_mean
    sigma = np.sqrt(input_bn.avg_var + input_bn.eps)
    gamma = input_bn.gamma.array
    beta = input_bn.beta.array

    for j, pixel in enumerate(orig_image):
        C_frac = 255 * (- beta[j] * sigma[j] / gamma[j] + mu[j])
        if gamma[j] >= 0:
            # x ≥ ⌈255 (- βσ/γ + μ)⌉ = C
            C = int(math.ceil(C_frac))
            orig = (pixel >= C)
            if orig:
                if not sol[j]:
                    #print((j, pixel, C - 1))
                    if not (0 <= C - 1 <= 255):
                        print(f"invalid input at {j}: {C - 1}")
                    perturbated_image[j] = C - 1
            else:
                if sol[j]:
                    #print((j, pixel, C))
                    if not (0 <= C <= 255):
                        print(f"invalid input at {j}: {C}")
                    perturbated_image[j] = C
        else:
            # x ≤ ⌊255 (- βσ/γ + μ)⌋ = C
            C = int(math.floor(C_frac))
            orig = (pixel <= C)
            if orig:
                if not sol[j]:
                    #print((j, pixel, C + 1))
                    if not (0 <= C + 1 <= 255):
                        print(f"invalid input at {j}: {C + 1}")
                    perturbated_image[j] = C + 1
            else:
                if sol[j]:
                    #print((j, pixel, C))
                    if not (0 <= C <= 255):
                        print(f"invalid input at {j}: {C}")
                    perturbated_image[j] = C

    return perturbated_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--format', type=str, choices=['maxsat', 'gurobi'], default='maxsat', help='solution format')
    parser.add_argument('--dataset', type=str, help='dataset name: mnist, mnist_back_image, mnist_rot')
    parser.add_argument('--instance', type=int, help='instance number')
    parser.add_argument('--output-image', '-o', type=str, default=None, help='output perturbated image')
    parser.add_argument('--output-orig-image', type=str, default=None, help='output original image')
    parser.add_argument('--output-diff-image', type=str, default=None, help='output diff image')
    parser.add_argument('--diff-image-scale', type=int, default=1, help='diff image scale')
    parser.add_argument('file', type=str, help='solution filename')
    args = parser.parse_args()

    if args.format == "maxsat":
        sol = read_binary_solution_maxsat(args.file)
    elif args.format == "gurobi":
        sol = read_binary_solution_gurobi(args.file)
    else:
        raise RuntimeError(f"unknown format: {args.format}")
    if sol is None:
        exit()

    train, test = datasets.get_dataset(args.dataset)

    neurons = [28 * 28, 200, 100, 100, 100, 10]
    model = bnn.BNN(neurons, stochastic_activation=True)
    chainer.serializers.load_npz(f"models/{args.dataset}.npz", model)

    scaled_image = test[args.instance][0]
    orig_image = np.round(scaled_image * 255).astype(np.uint8)
    perturbated_image = decode_binary_solution(model, orig_image, sol)

    print("original image:")
    with chainer.using_config("train", False), chainer.using_config("enable_backprop", True):
        logits = model((orig_image.astype(np.float32) / 255.0)[None])
        prob = F.softmax(logits)
    print(f"  logits: {list(logits.array[0])}")
    print(f"  probability: {list(prob.array[0])}")
    print(f"  predicted class: {np.argmax(logits.array[0])}")
    if args.output_orig_image is not None:
        img = visualize.to_image(args.dataset, orig_image)
        img.save(args.output_orig_image)

    print("perturbated image:")
    with chainer.using_config("train", False), chainer.using_config("enable_backprop", True):
        logits = model((perturbated_image.astype(np.float32) / 255.0)[None])
        prob = F.softmax(logits)
    print(f"  logits: {list(logits.array[0])}")
    print(f"  probability: {list(prob.array[0])}")
    print(f"  predicted class: {np.argmax(logits.array[0])}")
    if args.output_image is not None:
        img = visualize.to_image(args.dataset, perturbated_image)
        img.save(args.output_image)

    print("difference:")
    diff = perturbated_image.astype(np.int32) - orig_image.astype(np.int32)
    print(diff)
    for norm in [0, 1, 2, np.inf]:
        z = np.linalg.norm(diff, ord=norm)
        print(f"{norm}-norm: {z}")
    if args.output_diff_image is not None:
        img = visualize.to_image(args.dataset, (diff * args.diff_image_scale + 127).astype(np.uint8))
        img.save(args.output_diff_image)
