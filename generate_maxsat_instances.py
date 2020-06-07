import copy
import math
from pathlib import Path
from typing import Counter, Dict, List, Optional, Sequence, Tuple, Union

import chainer
import numpy as np

import bnn
import datasets
import encoder


result_dir = Path("instances/maxsat")

debug = False

debug_truelabel = False

debug_sat = False

produce_wbo = True

produce_wcnf = True

instances_per_class = 10

problems = [
    ("mnist", chainer.datasets.get_mnist, "models/mnist.npz"),
    ("mnist_back_image", datasets.get_mnist_back_image, "models/mnist_back_image.npz"),
    ("mnist_rot", datasets.get_mnist_rot, "models/mnist_rot.npz"),
]


def add_norm(enc: encoder.Encoder, norm: Union[int, str],
             mod: Sequence[Tuple[encoder.Lit, Optional[int]]]) -> None:
    if norm == 0:
        for lit, w in mod:
            enc.add_clause([lit], None if w is None else 1)
    elif norm == 1:
        for lit, w in mod:
            enc.add_clause([lit], None if w is None else abs(w))
    elif norm == 2:
        for lit, w in mod:
            enc.add_clause([lit], None if w is None else w*w)
    elif norm == 'inf':
        d: Dict[int, List[encoder.Lit]] = {}
        for lit, w in mod:
            if w is None:
                enc.add_clause([lit])
            else:
                w = abs(w)
                if w not in d:
                    d[w] = []
                d[w].append(lit)
        c_prev = 0
        relax_prev = None
        for w in sorted(d.keys()):
            relax = enc.new_var()
            enc.add_clause([-relax], cost=w - c_prev)
            if relax_prev is not None:
                enc.add_clause([-relax, relax_prev])  # relax → relax_prev
            for lit in d[w]:
                enc.add_clause([relax, lit])  # ¬lit → relax
            c_prev = w
            relax_prev = relax
    else:
        raise RuntimeError("unknown norm: " + str(norm))


result_dir.mkdir(parents=True, exist_ok=True)

encodings = \
    ([("wbo", "")] if produce_wbo else []) + \
    ([("wcnf", "parallel")] if produce_wcnf else [])
# , ("wcnf", "sequential")


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

    for ext, counter_encoding in encodings:
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

        for instance_no, (x, true_label) in enumerate(test):
            print(f"dataset={dataset_name}; instance={instance_no}; true_label={true_label} predicted_label={predicated_label[instance_no]}")
            if predicated_label[instance_no] != true_label:
                continue
            if counter[true_label] >= instances_per_class:
                continue
            counter[true_label] += 1

            with chainer.using_config("train", False), chainer.using_config("enable_backprop", True):
                saliency_map = model.saliency_map(x, true_label)

            for target in ["adversarial"] + (["truelabel"] if debug_truelabel else []):
                enc = copy.copy(enc_base)
                if target == "truelabel":
                    enc.add_clause([outputs[true_label]])
                else:
                    enc.add_clause([-outputs[true_label]])

                input_bn = model.input_bn
                mu = input_bn.avg_mean
                sigma = np.sqrt(input_bn.avg_var + input_bn.eps)
                gamma = input_bn.gamma.array
                beta = input_bn.beta.array

                mod: List[Tuple[encoder.Lit, Optional[int]]] = []
                for j, pixel in enumerate(orig_image[instance_no]):
                    C_frac = 255 * (- beta[j] * sigma[j] / gamma[j] + mu[j])
                    if gamma[j] >= 0:
                        # x ≥ ⌈255 (- βσ/γ + μ)⌉ = C
                        C = int(math.ceil(C_frac))
                        assert orig_image_bin[instance_no, j] == (pixel >= C)
                        if pixel < C:
                            mod.append((- inputs[j], C - pixel))
                        elif C == 0:
                            mod.append((inputs[j], None))  # impossible to change
                        else:
                            mod.append((inputs[j], (C - 1) - pixel))
                    else:
                        # x ≤ ⌊255 (- βσ/γ + μ)⌋ = C
                        C = int(math.floor(C_frac))
                        assert orig_image_bin[instance_no, j] == (pixel <= C)
                        if pixel > C:
                            mod.append((- inputs[j], C - pixel))
                        elif C == 255:
                            mod.append((inputs[j], None))  # impossible to change
                        else:
                            mod.append((inputs[j], (C + 1) - pixel))

                # debug
                if debug_sat:
                    enc2 = copy.copy(enc)
                    for lit, w in mod:
                        enc2.add_clause([lit])
                    if ext == "wcnf":
                        fname = result_dir / f"bnn_{dataset_name}_{instance_no}_label{true_label}_{target}_{counter_encoding}_debug.cnf"
                    elif ext == "wbo":
                        fname = result_dir / f"bnn_{dataset_name}_{instance_no}_label{true_label}_{target}_debug.opb"
                    else:
                        raise RuntimeError("unknown ext: " + ext)
                    enc2.write_to_file(fname)

                for ratio, ratio_str in [(1.0, ""), (0.2, "20p")]:  # , (0.5, "50p")
                    if ratio != 1.0:
                        important_pixels = set(list(reversed(np.argsort(np.abs(saliency_map))))[:int(len(saliency_map) * ratio)])
                        important_variables = set(inputs[instance_no] for i in important_pixels)
                        mod2 = [(lit, w if abs(lit) in important_variables else None) for lit, w in mod]
                    else:
                        mod2 = mod

                    norms: List[Union[int, str]] = [0, 1, 2, 'inf']
                    for norm in norms:
                        suffix = ''.join(["_" + s for s in (target, "norm_" + str(norm), ratio_str, counter_encoding) if len(s) > 0])
                        fname = result_dir / f"bnn_{dataset_name}_{instance_no}_label{true_label}{suffix}.{ext}"
                        enc2 = copy.copy(enc)
                        add_norm(enc2, norm, mod2)
                        enc2.write_to_file_opt(fname)

            if debug:
                break
