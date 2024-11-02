import argparse
import copy
import math
from pathlib import Path
from typing import Counter, Dict, List, Optional, Sequence, Tuple

import chainer
import numpy as np

import bnn
import datasets
import encoder_mip
import visualize


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['mnist', 'mnist_back_image', 'mnist_rot'], default='mnist', help='dataset name')
parser.add_argument('--model', type=str, default=None, help='model file (*.npz)')
parser.add_argument('-o', '--output-dir', type=str, default="instances/mip", help='output directory')
parser.add_argument('--norm', type=str, choices=['0', '1', '2', 'inf'], nargs='*', default=['0', '1', '2', 'inf'], help='norm to minimize')
parser.add_argument('--use-indicator', default=False, action="store_true", help='use indicator constraints')
parser.add_argument('--target', type=str, default="adversarial", choices=['adversarial', 'truelabel'], help='target label')
parser.add_argument('--instance-no', type=int, nargs='*', default=[], help='specify instance number')
parser.add_argument('--instances-per-class', type=int, default=None, help='number of instances to generate per class')
parser.add_argument('--debug-sat', action='store_true', help='produce CNF or OPB for debug')
parser.add_argument('--ratio', type=float, nargs='*', default=[1.0], help='restrict search space to most salient pixels')


args = parser.parse_args()

result_dir = Path(args.output_dir)
result_dir.mkdir(parents=True, exist_ok=True)

train, test = datasets.get_dataset(args.dataset)

if args.model is None:
    weights_filename = f"models/{args.dataset}.npz"
else:
    weights_filename = args.model
neurons = [28 * 28, 200, 100, 100, 100, 10]
model = bnn.BNN(neurons, stochastic_activation=True)
chainer.serializers.load_npz(weights_filename, model)

orig_image_scaled = test._datasets[0]
orig_image_scaled = orig_image_scaled[:1000]
orig_image = np.round(orig_image_scaled * 255).astype(np.uint8)
with chainer.using_config("train", False), chainer.using_config("enable_backprop", False):
    orig_image_bin = bnn.bin(model.input_bn(orig_image_scaled)).array > 0
    orig_logits = model(orig_image_scaled).array
predicated_label = np.argmax(orig_logits, axis=1)

enc_base = encoder_mip.BNNEncoder(args.use_indicator)

# input = [self.new_var(f"input_{i}", 0, 255, True) for i in range(28*28)]
inputs = [enc_base.new_var(f"input_bin({i})", 0, 1, True) for i in range(28*28)]
outputs = [enc_base.new_var(f"output({i})", 0, 1, True) for i in range(10)]
enc_base.encode_bin_input(model, inputs, outputs)

counter = Counter[int]()

for instance_no, (x, true_label) in enumerate(test):
    if len(args.instance_no) > 0 and instance_no not in args.instance_no:
        continue
    if args.instances_per_class is not None and counter[true_label] >= args.instances_per_class:
        continue
    print(f"dataset={args.dataset}; instance={instance_no}; true_label={true_label} predicted_label={predicated_label[instance_no]}")
    if predicated_label[instance_no] != true_label:
        continue
    counter[true_label] += 1

    img = visualize.to_image(args.dataset, orig_image[instance_no])
    fname = result_dir / f"bnn_{args.dataset}_{instance_no}_label{true_label}.png"
    if not fname.exists():
        img.save(fname)

    with chainer.using_config("train", False), chainer.using_config("enable_backprop", True):
        saliency_map = model.saliency_map(x, true_label)

    enc = copy.copy(enc_base)
    if args.target == "truelabel":
        enc.add([(1, outputs[true_label])], ">=", 1)
    else:
        enc.add([(1, outputs[true_label])], "<=", 0)

    input_bn = model.input_bn
    mu = input_bn.avg_mean
    sigma = np.sqrt(input_bn.avg_var + input_bn.eps)
    gamma = input_bn.gamma.array
    beta = input_bn.beta.array

    numerically_unstable = False

    mod: List[Tuple[encoder_mip.Var, bool, Optional[int]]] = []
    for j, pixel in enumerate(orig_image[instance_no]):
        # C_frac = 255 * (- beta[j] * sigma[j] / gamma[j] + mu[j])
        C_frac = (- beta[j] * sigma[j] / gamma[j] + mu[j]) / np.float32(1 / 255.0)
        if gamma[j] >= 0:
            # x ≥ ⌈255 (- βσ/γ + μ)⌉ = C
            C = int(math.ceil(C_frac))
            if orig_image_bin[instance_no, j] != (pixel >= C):
                numerically_unstable = True
                break
            # assert orig_image_bin[instance_no, j] == (pixel >= C)
            if pixel < C:
                if C <= 255:
                    mod.append((inputs[j], True, C - pixel))
                else:
                    mod.append((inputs[j], True, None))  # impossible to change
            else:
                if C - 1 >= 0:
                    mod.append((inputs[j], False, (C - 1) - pixel))
                else:
                    mod.append((inputs[j], False, None))  # impossible to change
        else:
            # x ≤ ⌊255 (- βσ/γ + μ)⌋ = C
            C = int(math.floor(C_frac))
            # assert orig_image_bin[instance_no, j] == (pixel <= C)
            if orig_image_bin[instance_no, j] != (pixel <= C):
                numerically_unstable = True
                break
            if pixel > C:
                if C >= 0:
                    mod.append((inputs[j], True, C - pixel))
                else:
                    mod.append((inputs[j], True, None))
            else:
                if C + 1 <= 255:
                    mod.append((inputs[j], False, (C + 1) - pixel))
                else:
                    mod.append((inputs[j], False, None))  # impossible to change

    if numerically_unstable:
        print("numerically unstable")
        continue

    for ratio in args.ratio:
        if ratio == 1.0:
            mod2 = mod
            ratio_str = ""
        else:
            ratio_str = f"{int(ratio * 100)}p"
            important_pixels = set(list(reversed(np.argsort(np.abs(saliency_map))))[:int(len(saliency_map) * ratio)])
            important_variables = set(inputs[instance_no] for i in important_pixels)
            mod2 = [(var, b, w if var in important_variables else None) for var, b, w in mod]

        for norm in args.norm:
            xs: List[str] = [
                "bnn", args.dataset, str(instance_no), f"label{true_label}",
                args.target, "norm_" + str(norm), ratio_str,
                "indicator" if args.use_indicator else ''
            ]
            fname = result_dir / ('_'.join([s for s in xs if len(s) > 0]) + ".lp")
            enc2 = copy.copy(enc)
            enc2.set_norm_objective(norm, mod2)
            enc2.write_to_file(fname)
