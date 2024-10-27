import copy
import math
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

import bnn


Var = str

Term = Tuple[int, Var]

Expr = List[Term]


def show_expr(e: Expr) -> str:
    tmp = []
    for c, v in e:
        if c == 1:
            tmp.append(f"+ {v}")
        elif c == -1:
            tmp.append(f"- {v}")
        elif c >= 0:
            tmp.append(f"+ {c} {v}")
        else:
            tmp.append(f"- {abs(c)} {v}")
    return " ".join(tmp)


class Constr(NamedTuple):
    lhs: Expr
    op: str
    rhs: int
    indicator: Optional[Tuple[Var, int]]

    def __str__(self) -> str:
        if self.indicator is None:
            return f"{show_expr(self.lhs)} {self.op} {self.rhs}"
        else:
            return f"{self.indicator[0]} = {self.indicator[1]} -> {show_expr(self.lhs)} {self.op} {self.rhs}"


class Encoder():
    def __init__(self) -> None:
        self.vars: Dict[Var, Tuple[Optional[float], Optional[float], bool]] = {}
        self.constrs: List[Constr] = []
        self.objective: Expr = []

    def __copy__(self):
        ret = self.__class__()
        ret.vars = copy.copy(self.vars)
        ret.constrs = copy.copy(self.constrs)
        ret.objective = copy.copy(self.objective)
        return ret

    def new_var(self, name: str, lb: Optional[float], ub: Optional[float], is_int: bool = False) -> Var:
        self.vars[name] = (lb, ub, is_int)
        return name

    def add(self, lhs: Expr, op: str, rhs: int) -> None:
        self.constrs.append(Constr(lhs, op, rhs, None))

    def add_indicator(self, indicator_lhs: Var, indicator_rhs: int, lhs: Expr, op: str, rhs: int) -> None:
        self.constrs.append(Constr(lhs, op, rhs, (indicator_lhs, indicator_rhs)))

    def set_objective(self, obj: Expr) -> None:
        self.objective = obj

    def write_to_file(self, filename: Union[str, Path]) -> None:
        bins = set()
        gens = set()

        for x, (lb, ub, is_int) in self.vars.items():
            if is_int:
                if lb == 0 and ub == 1:
                    bins.add(x)
                else:
                    gens.add(x)

        with open(filename, "w") as f:
            f.write("MINIMIZE\n")
            if len(self.objective) == 0:
                f.write("+0 dummy\n")
            else:
                f.write(show_expr(self.objective) + "\n")
            f.write("SUBJECT TO\n")
            for constr in self.constrs:
                f.write(f"{constr}\n")
            f.write("BOUNDS\n")
            if len(self.objective) == 0:
                f.write("0 <= dummy <= 0\n")
            for x, (lb, ub, is_int) in self.vars.items():
                if x in bins:
                    continue
                if lb is None:
                    lb2 = "-inf"
                else:
                    lb2 = str(lb)
                if ub is None:
                    ub2 = "+inf"
                else:
                    ub2 = str(ub)
                f.write(f"{lb2} <= {x} <= {ub2}\n")
            if len(bins) > 0:
                f.write("BINARIES\n")
                f.write(" ".join(sorted(bins)))
                f.write("\n")
            if len(gens) > 0:
                f.write("GENERALS\n")
                f.write(" ".join(sorted(gens)))
                f.write("\n")
            f.write("END\n")


class BNNEncoder(Encoder):

    def __init__(self, use_indicator: bool = False) -> None:
        super().__init__()
        self.use_indicator = use_indicator

    def add_ge_soft(self, y: Var, lhs: Expr, rhs: int) -> None:
        # y → lhs ≥ rhs
        lb = sum(c * (self.vars[x][0] if c >= 0 else self.vars[x][1]) for c, x in lhs)
        if self.use_indicator:
            self.add_indicator(y, 1, lhs, '>=', rhs)
        else:
            # y → lhs ≥ rhs
            # M1 (1-y) + lhs ≥ rhs where M1 = rhs - lb
            # -M1 y + lhs ≥ rhs - M1
            M1 = rhs - lb
            self.add([(-M1, y)] + lhs, '>=', rhs - M1)

    def encode_atleast(self, y: Var, lhs: Expr, rhs: int) -> None:
        # y ↔ lhs ≥ rhs
        lb = sum(c * (self.vars[x][0] if c >= 0 else self.vars[x][1]) for c, x in lhs)
        ub = sum(c * (self.vars[x][1] if c >= 0 else self.vars[x][0]) for c, x in lhs)

        # y → lhs ≥ rhs
        if self.use_indicator:
            self.add_indicator(y, 1, lhs, '>=', rhs)
        else:
            # M1 (1-y) + lhs ≥ rhs where M1 = rhs - lb
            # -M1 y + lhs ≥ rhs - M1
            M1 = rhs - lb
            self.add([(-M1, y)] + lhs, '>=', rhs - M1)

        # ¬y → ¬ (lhs ≥ rhs)
        # ¬y → lhs < rhs
        # ¬y → lhs ≤ rhs - 1
        if self.use_indicator:
            self.add_indicator(y, 0, lhs, '<=', rhs - 1)
        else:
            # M2 y + lhs ≤ rhs - 1 where M2 = (rhs - 1) - ub
            M2 = (rhs - 1) - ub
            self.add([(M2, y)] + lhs, '<=', rhs - 1)

    def encode_binarize_pixel(self, mu: float, sigma: float, gamma: float, beta: float, x: Var, y: Var) -> None:
        # γ((x/255 - μ) / σ) + β ≥ 0
        # (γ/σ) (x/255 - μ) ≥ - β
        C_frac = 255 * (- beta * sigma / gamma + mu)
        if gamma >= 0:
            # x ≥ ⌈255 (- βσ/γ + μ)⌉ = C
            C = int(math.ceil(C_frac))
            self.encode_atleast(y, [(1, x)], C)
        else:
            # x ≤ ⌊255 (- βσ/γ + μ)⌋ = C
            C = int(math.floor(C_frac))
            self.encode_atleast(y, [(-1, x)], -C)

    def encode_binarize_image(self, input_bn, xs: Sequence[Var], ys: Sequence[Var]) -> None:
        assert len(xs) == len(ys)
        mu = input_bn.avg_mean
        sigma = np.sqrt(input_bn.avg_var + input_bn.eps)
        gamma = input_bn.gamma.data
        beta = input_bn.beta.data
        for i in range(len(xs)):
            self.encode_binarize_pixel(mu[i], sigma[i], gamma[i], beta[i], xs[i], ys[i])

    def encode_block_1(self, row, b: float, mu: float, sigma: float, gamma: float, beta: float, xs: Sequence[Var], y: Var) -> None:
        # assert all(int(x) == 1 or int(x) == -1 for x in row.reshape(-1))
        # γ((Σ_i row[i]*(2*xs[i]-1) + b) - μ)/σ + β ≥ 0
        # ((Σ_i row[i]*(2*xs[i]-1) + b) - μ)*(γ/σ) ≥ -β
        C_frac = (- beta * sigma / gamma + mu - b + sum(row)) / 2
        if gamma >= 0:
            # Σ_i row[i]*xs[i] ≥ ⌈(-βσ/γ + μ - b + Σ_i row[i]) / 2⌉ = C
            C = int(math.ceil(C_frac))
            self.encode_atleast(y, [(int(row[i]), x) for (i, x) in enumerate(xs)],  C)
        else:
            # Σ_i row[i]*xs[i] ≤ ⌊(-βσ/γ + μ - b + Σ_i row[i]) / 2⌋ = C
            C = int(math.floor(C_frac))
            self.encode_atleast(y, [(-int(row[i]), x) for (i, x) in enumerate(xs)], -C)

    def encode_block(self, block: bnn.Block, no: int, xs: Sequence[Var]) -> List[Var]:
        n_output, n_input = block.lin.W.shape
        assert len(xs) == n_input
        W = block.lin.W.data.astype(np.int32)
        b = block.lin.b.data
        mu = block.bn.avg_mean
        sigma = np.sqrt(block.bn.avg_var + block.bn.eps)
        gamma = block.bn.gamma.data
        beta = block.bn.beta.data
        assert all(int(x) == 1 or int(x) == -1 for x in W.astype(np.int32).reshape(-1))
        output = [self.new_var(f"block{no}({i})", 0, 1, True) for i in range(n_output)]
        for i in range(n_output):
            self.encode_block_1(W[i].astype(np.int32), b[i], mu[i], sigma[i], gamma[i], beta[i], xs, output[i])
        return output

    def encode_output(self, lin: bnn.BinLinear, xs: Sequence[Var], output: Sequence[Var]) -> None:
        m = len(xs)
        n = len(output)
        self.add([(1, x) for x in output], '=', 1)

        W = lin.W.data.astype(np.int32)
        b = lin.b.data
        assert W.shape == (n, m)
        assert b.shape == (n, )
        #   (Σ_j W[i,j] (2xs[j]-1)) + b[i]
        # = (Σ_j 2 W[i,j] xs[j]) - (Σ_j W[i,j]) + b[i]
        logits = [(2 * W[i, :], - sum(W[i, :]) + b[i]) for i in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # logits[i] ≥ logits[j]
                lhs = [(w, xs[k]) for k, w in enumerate(logits[i][0] - logits[j][0]) if w != 0]
                rhs = int(math.ceil(logits[j][1] - logits[i][1]))
                self.add_ge_soft(output[i], lhs, rhs)

    def encode(self, model: bnn.BNN, image: Sequence[Var], output: Sequence[Var]) -> None:
        input_bin = [self.new_var(f"input_bin({i})", 0, 1, True) for i in range(28*28)]
        self.encode_binarize_image(model.input_bn, image, input_bin)
        encode_bin_input(model, input_bin, output)

    def encode_bin_input(self, model: bnn.BNN, image: Sequence[Var], output: Sequence[Var]) -> None:
        h = image
        for i, block in enumerate(model.blocks):
            h = self.encode_block(block, i, h)
        self.encode_output(model.output_lin, h, output)

    def set_norm_objective(self, norm: str, mod: Sequence[Tuple[Var, bool, Optional[int]]]) -> None:
        def f(w: int) -> int:
            if norm == '0':
                w2 = 1
            elif norm == '1':
                w2 = abs(w)
            elif norm == '2':
                w2 = w * w
            else:
                assert False
            return w2

        if norm == "inf":
            top = self.new_var("top", 0, None, is_int=False)

            for var, b, w in mod:
                if b:
                    # The pixel is 0 in the original image
                    if w is None:
                        # impossible to flip
                        self.add([(1, var)], "<=", 0)
                    else:
                        # |w| var <= top
                        self.add([(abs(w), var), (-1, top)], "<=", 0)
                else:
                    # The pixel is 1 in the original image
                    if w is None:
                        self.add([(1, var)], ">=", 1)
                    else:
                        # |w| (1 - var) <= top
                        self.add([(- abs(w), var), (-1, top)], "<=", - abs(w))

            self.set_objective([(1, top)])
        else:
            obj: Expr = []
            offset: int = 0

            for var, b, w in mod:
                if b:
                    # The pixel is 0 in the original image
                    if w is None:
                        # impossible to flip
                        self.add([(1, var)], "<=", 0)
                    else:
                        obj.append((f(w), var))
                else:
                    # The pixel is 1 in the original image
                    if w is None:
                        # impossible to flip
                        self.add([(1, var)], ">=", 1)
                    else:
                        obj.append((- f(w), var))
                        offset += f(w)

            if offset != 0:
                one = self.new_var("one", 1, 1, is_int=True)
                obj.append((offset, one))
            self.set_objective(obj)
