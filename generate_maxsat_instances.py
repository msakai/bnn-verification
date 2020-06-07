import copy
import math
from pathlib import Path
import re
import subprocess
from typing import Counter, Dict, List, Optional, Sequence, Tuple, Union

import chainer
import numpy as np

import bnn
import datasets
import encoder


result_dir = Path("instances/maxsat")
result_dir.mkdir(parents=True, exist_ok=True)

problems = [
    ("mnist", chainer.datasets.get_mnist, "models/mnist.npz"),
    ("mnist_back_image", datasets.get_mnist_back_image, "models/mnist_back_image.npz"),
    ("mnist_rot", datasets.get_mnist_rot, "models/mnist_rot.npz"),
]


def add_norm(enc: encoder.Encoder, norm: Union[int, str], mod: Sequence[Tuple[encoder.Lit, Optional[int]]]) -> None:
    if norm == 0:
        for l, w in mod:
            enc.add_clause([l], None if w is None else 1)
    elif norm == 1:
        for l, w in mod:
            enc.add_clause([l], None if w is None else abs(w))
    elif norm == 2:
        for l, w in mod:
            enc.add_clause([l], None if w is None else w*w)
    elif norm == 'inf':
        d: Dict[int, List[encoder.Lit]] = {}
        for l, w in mod:
            if w is None:
                enc.add_clause([l])
            else:
                w = abs(w)
                if w not in d:
                    d[w] = []
                d[w].append(l)
        c_prev = 0
        relax_prev = None
        for w in sorted(d.keys()):
            relax = enc.new_var()
            enc.add_clause([-relax], cost = w - c_prev)
            if relax_prev is not None:
                enc.add_clause([-relax, relax_prev])  # relax → relax_prev
            for l in d[w]:
                enc.add_clause([relax, l]) # ¬l → relax
            c_prev = w
            relax_prev = relax        
    else:
        raise RuntimeError("unknown norm: " + str(norm))
    

for dataset_name, get_dataset, weights_filename in problems:
    train, test = get_dataset()

    neurons = [28 * 28, 200, 100, 100, 100, 10]
    model = bnn.BNN(neurons, stochastic_activation=True)
    chainer.serializers.load_npz(weights_filename, model)

    orig_image_scaled = test._datasets[0]
    orig_image = (orig_image_scaled * 255).astype(np.int)
    with chainer.using_config("train", False), chainer.using_config("enable_backprop", False):
        orig_image_bin = bnn.bin(model.input_bn(orig_image_scaled)).array > 0
        orig_logits = model(orig_image_scaled).array
    predicated_label = np.argmax(orig_logits, axis=1)

    for ext, counter_encoding in [("wbo", ""), ("wcnf", "parallel")]: # , ("wcnf", "sequential")
        if ext == "wbo":
            enc_base = encoder.BNNEncoder(cnf=False)
        elif ext == "wcnf":
            enc_base = encoder.BNNEncoder(cnf=True, counter=counter_encoding)
        else:
            raise RuntimeError("unknown ext: " + ext)
        inputs = enc_base.new_vars(784)
        outputs = enc_base.new_vars(10)
        enc_base.encode_bin_input(model, inputs, outputs)

        counter = Counter[int]()

        for i, (x, true_label) in enumerate(test):
            print(f"dataset={dataset_name}; instance={i}; true_label={true_label} predicted_label={predicated_label[i]}")
            if predicated_label[i] != true_label:
                continue
            if counter[true_label] > 0:
                continue
            counter[true_label] += 1

            with chainer.using_config("train", False), chainer.using_config("enable_backprop", True):
                saliency_map = model.saliency_map(x, true_label)

            for label in ["true", "adv"]:
                enc = copy.copy(enc_base)
                if label == "true":
                    enc.add_clause([outputs[true_label]])
                else:
                    enc.add_clause([-outputs[true_label]])

                input_bn = model.input_bn
                mu = input_bn.avg_mean
                sigma = np.sqrt(input_bn.avg_var + input_bn.eps)
                gamma = input_bn.gamma.array
                beta = input_bn.beta.array

                mod: List[Tuple[encoder.Lit, Optional[int]]] = []
                for j, pixel in enumerate(orig_image[i]):
                    C_frac = 255 * (- beta[j] * sigma[j] / gamma[j] + mu[j])
                    if gamma[j] >= 0:
                        # x ≥ ⌈255 (- βσ/γ + μ)⌉ = C
                        C = int(math.ceil(C_frac))
                        assert orig_image_bin[i,j] == (pixel >= C)
                        if pixel < C:
                            mod.append((- inputs[j], C - pixel))
                        elif C == 0:
                            mod.append((inputs[j], None)) # impossible to change
                        else:
                            mod.append((inputs[j], (C - 1) - pixel))
                    else:
                        # x ≤ ⌊255 (- βσ/γ + μ)⌋ = C
                        C = int(math.floor(C_frac))
                        assert orig_image_bin[i,j] == (pixel <= C)
                        if pixel > C:
                            mod.append((- inputs[j], C - pixel))
                        elif C == 255:
                            mod.append((inputs[j], None)) # impossible to change
                        else:
                            mod.append((inputs[j], (C + 1) - pixel))

                # debug
                enc2 = copy.copy(enc)
                for l, w in mod:
                    enc2.add_clause([l])
                if ext == "wcnf":
                    fname = result_dir / f"{dataset_name}_test{i}_{label}_{counter_encoding}_debug.cnf"
                elif ext == "wbo":
                    fname = result_dir / f"{dataset_name}_test{i}_{label}_debug.opb"
                else:
                    raise RuntimeError("unknown ext: " + ext)
                enc2.write_to_file(fname)

                for ratio, ratio_str in [(1.0, ""), (0.2, "20p")]: # , (0.5, "50p")
                    if ratio != 1.0:
                        important_pixels = set(list(reversed(np.argsort(np.abs(saliency_map))))[:int(len(saliency_map) * ratio)])
                        important_variables = set(inputs[i] for i in important_pixels)
                        mod2 = [(lit, w if abs(lit) in important_variables else None) for lit, w in mod]
                    else:
                        mod2 = mod

                    norms: List[Union[int, str]] = [0, 1, 2, 'inf']
                    for norm in norms:
                        suffix = ''.join(["_" + s for s in (str(norm), ratio_str, counter_encoding) if len(s) > 0])
                        fname = result_dir / f"{dataset_name}_test{i}_{label}{suffix}.{ext}"
                        enc2 = copy.copy(enc)
                        add_norm(enc2, norm, mod2)
                        enc2.write_to_file_opt(fname)

            break
