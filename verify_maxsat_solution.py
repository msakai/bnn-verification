import argparse
import math
from pathlib import Path
import re
from typing import Optional, Sequence, Union

import chainer
import numpy as np
import PIL

import bnn
import datasets


def read_binary_solution(fname: Union[Path, str]) -> Optional[np.ndarray]:
    sol = np.zeros(28*28, dtype=bool)
    with open(fname) as f:
        for line in f:
            if line.startswith('v '):
                for s in line[2:].split():
                    l = int(s)
                    if abs(l) <= 28*28:
                        sol[abs(l) - 1] = (l > 0)
    return sol


def decode_binary_solution(model, orig_image: np.ndarray, sol: np.ndarray) -> np.ndarray:
    pertubated_image = orig_image.copy()

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
                    pertubated_image[j] = C - 1
            else:
                if sol[j]:
                    #print((j, pixel, C))
                    pertubated_image[j] = C
        else:
            # x ≤ ⌊255 (- βσ/γ + μ)⌋ = C
            C = int(math.floor(C_frac))
            orig = (pixel <= C)
            if orig:
                if not sol[j]:
                    #print((j, pixel, C + 1))
                    pertubated_image[j] = C + 1
            else:
                if sol[j]:
                    #print((j, pixel, C))
                    pertubated_image[j] = C

    return pertubated_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset name: mnist, mnist_back_image, mnist_rot')
    parser.add_argument('--instance', type=int, help='instance number')
    parser.add_argument('--output-image', '-o', type=str, default=None, help='output image')
    parser.add_argument('file', type=str, help='solution filename')
    args = parser.parse_args()

    sol = read_binary_solution(args.file)
    if sol is None:
        exit()

    train, test = datasets.get_dataset(args.dataset)

    neurons = [28 * 28, 200, 100, 100, 100, 10]
    model = bnn.BNN(neurons, stochastic_activation=True)
    chainer.serializers.load_npz(f"models/{args.dataset}.npz", model)

    scaled_image = test[args.instance][0]
    orig_image = np.round(scaled_image * 255).astype(np.uint8)
    pertubated_image = decode_binary_solution(model, orig_image, sol)

    diff = pertubated_image.astype(np.int32) - orig_image.astype(np.int32)
    print(diff)

    for norm in [0, 1, 2, np.inf]:
        z = np.linalg.norm(diff, ord=norm)
        print(f"{norm}-norm: {z}")

    with chainer.using_config("train", False), chainer.using_config("enable_backprop", True):
        logits = model((pertubated_image.astype(np.float32) / 255.0)[None]).array[0]
    print(f"predicted class: {np.argmax(logits)}")
    print(f"logits: {list(logits)}")

    if args.output_image is not None:
        img = PIL.Image.fromarray(pertubated_image.reshape(28, 28))
        img.save(args.output_image)
