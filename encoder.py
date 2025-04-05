import copy
import math
from pathlib import Path
from typing import Counter, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

import bnn


Var = int

Lit = int

lit_true: Lit = 0x7fffffff

lit_false: Lit = -lit_true

PBSum = List[Tuple[int, Lit]]


class Polarity(NamedTuple):
    pos: bool
    neg: bool

    def negate(self) -> 'Polarity':
        return Polarity(self.neg, self.pos)


def normalize_clause(clause: Sequence[Lit]) -> Optional[List[Lit]]:
    if lit_true in clause:
        return None
    else:
        return [lit for lit in clause if lit != lit_false]


def normalize_pbsum(xs: PBSum) -> Tuple[PBSum, int]:
    def f(xs: PBSum, offset: int) -> Tuple[PBSum, int]:
        xs2 = Counter[Lit]()
        for (c, l) in xs:
            if l > 0:
                xs2[l] += c
            else:
                # c ¬x = c (1 - x) = c - c x
                xs2[-l] -= c
                offset += c
        offset += xs2.get(lit_true, 0)
        del xs2[lit_true]
        return ([(c, l) for (l, c) in xs2.items() if c != 0], offset)

    def g(xs: PBSum, offset: int) -> Tuple[PBSum, int]:
        xs2 = []
        for (c, x) in xs:
            if c >= 0:
                xs2.append((c, x))
            else:
                # c x = -c (1 - x) + c = -c ¬x + c
                xs2.append((- c, -x))
                offset += c
        return PBAtLeast(xs2, offset)

    return g(*f(xs, 0))


def show_pbsum(xs: PBSum) -> str:
    return " ".join((f"+{c} " if c >= 0 else f"{c} ") + (f"x{v}" if v >= 0 else f"~x{-v}") for c, v in xs)


class PBAtLeast(NamedTuple):
    lhs: PBSum
    rhs: int

    def __str__(self) -> str:
        return f"{show_pbsum(self.lhs)} >= {self.rhs}"

    @property
    def op(self) -> str:
        return '>='

    # ¬ (lhs ≥ rhs) ⇔ (lhs < rhs) ⇔ (-lhs > -rhs) ⇔ (-lhs ≥ -rhs + 1)
    def negate(self) -> 'PBAtLeast':
        return PBAtLeast([(-c, x) for (c, x) in self.lhs], -self.rhs + 1)

    def to_pos_literals(self) -> 'PBAtLeast':
        lhs2 = []
        offset = 0
        for (c, l) in self.lhs:
            if l > 0:
                lhs2.append((c, l))
            else:
                # c ¬x = c (1 - x) = c - c x
                lhs2.append((-c, -l))
                offset += c
        return PBAtLeast(lhs2, self.rhs - offset)

    def normalize(self) -> 'PBAtLeast':
        lhs2, offset = normalize_pbsum(self.lhs)
        x = PBAtLeast(lhs2, self.rhs - offset)
        x = x._normalize_pb_atleast_trivial()
        x = x._normalize_pb_atleast_gcd()
        return x

    def _normalize_pb_atleast_trivial(self) -> 'PBAtLeast':
        if self.rhs <= 0:
            return PBAtLeast([], 0)
        elif sum(c for (c, _) in self.lhs) < self.rhs:
            return PBAtLeast([], 1)
        else:
            return self

    def _normalize_pb_atleast_gcd(self) -> 'PBAtLeast':
        if self.rhs <= 0:
            return self
        lhs = [(min(c, self.rhs), x) for (c, x) in self.lhs]

        g = None
        for (c, x) in lhs:
            assert c >= 0
            if c < self.rhs:
                if g is None:
                    g = c
                else:
                    g = math.gcd(g, c)

        if g is None:  # all coefficients are >=rhs
            lhs = [(1, x) for (c, x) in lhs]
            rhs = 1
        else:
            lhs = [((c + g - 1) // g, x) for (c, x) in lhs]
            rhs = (self.rhs + g - 1) // g
        return PBAtLeast(lhs, rhs)

    def fix_literals(self) -> Tuple['PBAtLeast', List[Lit]]:
        lhs = sorted(self.lhs, key=lambda x: x[0], reverse=True)
        slack = sum(c for (c, x) in lhs) - self.rhs
        lhs2 = []
        rhs2 = self.rhs
        fixed = []
        for i in range(len(lhs)):
            c, x = lhs[i]
            if c > slack:
                fixed.append(x)
                rhs2 -= c
            else:
                lhs2.append((c, x))
        return (PBAtLeast(lhs2, rhs2), fixed)


class PBExactly(NamedTuple):
    lhs: PBSum
    rhs: int

    def __str__(self) -> str:
        return f"{show_pbsum(self.lhs)} = {self.rhs}"

    @property
    def op(self) -> str:
        return '='

    def normalize(self) -> 'PBExactly':
        lhs2, offset = normalize_pbsum(self.lhs)
        x = PBExactly(lhs2, self.rhs - offset)
        return x

    def to_pos_literals(self) -> 'PBExactly':
        lhs2 = []
        offset = 0
        for (c, l) in self.lhs:
            if l > 0:
                lhs2.append((c, l))
            else:
                # c ¬x = c (1 - x) = c - c x
                lhs2.append((-c, -l))
                offset += c
        return PBExactly(lhs2, self.rhs - offset)


class ConjEntry:
    def __init__(self, var: Var) -> None:
        self.var = var
        self.pos_asserted = False
        self.neg_asserted = False


class Encoder():
    def __init__(self, cnf=True, counter="parallel") -> None:
        self._cnf = cnf
        self._counter = counter
        self.nvars = 0
        self.constrs: List[Tuple[Optional[int], Union[PBExactly, PBAtLeast]]] = []
        self.conj_table: Dict[Tuple[Lit, ...], ConjEntry] = {}

    def __copy__(self):
        ret = self.__class__(self._cnf, self._counter)
        ret.nvars = self.nvars
        ret.constrs = copy.copy(self.constrs)
        ret.conj_table = copy.deepcopy(self.conj_table)
        return ret

    def new_var(self) -> Lit:
        self.nvars += 1
        return self.nvars

    def new_vars(self, n: int) -> List[Lit]:
        return [self.new_var() for _ in range(n)]

    def add_clause(self, lits: Sequence[Lit], cost: Optional[int] = None) -> None:
        lits2 = normalize_clause(lits)
        if lits2 is not None:
            self.add_pb_atleast(PBAtLeast([(1, l) for l in lits2], 1), cost)

    def add_pb_atleast(self, constr: PBAtLeast, cost: Optional[int] = None) -> None:
        constr = constr.normalize()
        if self._lb(constr.lhs) >= constr.rhs:
            return
        self.constrs.append((cost, constr))

    def add_pb_atleast_soft(self, sel: Lit, constr: PBAtLeast) -> None:
        constr = constr.normalize()
        constr = PBAtLeast([(constr.rhs - self._lb(constr.lhs), -sel)] + constr.lhs, constr.rhs)
        self.add_pb_atleast(constr)

    def _lb(self, s: PBSum) -> int:
        return sum(c if c < 0 else 0 for (c, _) in s)

    def _ub(self, s: PBSum) -> int:
        return sum(c if c > 0 else 0 for (c, _) in s)

    def add_pb_exactly(self, constr: PBExactly, cost: Optional[int] = None) -> None:
        self.constrs.append((cost, constr.normalize()))

    def encode_conj(self, *xs: Lit, polarity: Polarity = Polarity(True, True)) -> Lit:
        if any(x == lit_false for x in xs):
            return lit_false
        else:
            xs = tuple(sorted([x for x in xs if x != lit_true]))
            if len(xs) == 0:
                return lit_true
            elif len(xs) == 1:
                return xs[0]
            else:
                if xs in self.conj_table:
                    e = self.conj_table[xs]
                else:
                    e = self.conj_table[xs] = ConjEntry(self.new_var())
                r = e.var
                if polarity.pos and not e.pos_asserted:
                    for x in xs:
                        self.add_clause([-r, x])
                    e.pos_asserted = True
                if polarity.neg and not e.neg_asserted:
                    self.add_clause([-x for x in xs] + [r])
                    e.neg_asserted = True
                return r

    def encode_disj(self, *xs: Lit, polarity: Polarity = Polarity(True, True)) -> Lit:
        return - self.encode_conj(*[-x for x in xs], polarity=polarity.negate())

    def encode_bvuge(self, lhs: Sequence[Lit], rhs: Sequence[Lit], polarity: Polarity = Polarity(True, True)) -> Lit:
        # lhs occurs positively and rhs occurs negatively
        ret = lit_true
        assert len(lhs) == len(rhs)
        for i in range(len(lhs)):
            cond1 = self.encode_conj(lhs[i], -rhs[i], polarity=polarity)
            cond2 = self.encode_conj(self.encode_disj(-rhs[i], lhs[i], polarity=polarity), ret, polarity=polarity)
            ret = self.encode_disj(cond1, cond2, polarity=polarity)
        return ret

    def encode_fa_sum(self, a, b, c, polarity: Polarity = Polarity(True, True)) -> Lit:
        x = self.new_var()
        if polarity.neg:
            # FASum(a,b,c) → x
            self.add_clause([-a, -b, -c, x])  #  a ∧  b ∧  c → x
            self.add_clause([-a, b, c, x])    #  a ∧ ¬b ∧ ¬c → x
            self.add_clause([a, -b, c, x])    # ¬a ∧  b ∧ ¬c → x
            self.add_clause([a, b, -c, x])    # ¬a ∧ ¬b ∧  c → x
        if polarity.pos:
            # x → FASum(a,b,c)
            # ⇔ ¬FASum(a,b,c) → ¬x
            self.add_clause([a, b, c, -x])    # ¬a ∧ ¬b ∧ ¬c → ¬x
            self.add_clause([a, -b, -c, -x])  # ¬a ∧  b ∧  c → ¬x
            self.add_clause([-a, b, -c, -x])  #  a ∧ ¬b ∧  c → ¬x
            self.add_clause([-a, -b, c, -x])  #  a ∧  b ∧ ¬c → ¬x
        return x

    def encode_fa_carry(self, a, b, c, polarity: Polarity = Polarity(True, True)) -> Lit:
        x = self.new_var()
        if polarity.pos:
            # x → FACarry(a,b,c)
            # ⇔ ¬FACarry(a,b,c) → ¬x
            self.add_clause([a, b, -x])  #  ¬a ∧ ¬b → ¬x
            self.add_clause([a, c, -x])  #  ¬a ∧ ¬c → ¬x
            self.add_clause([b, c, -x])  #  ¬b ∧ ¬c → ¬x
        if polarity.neg:
            # FACarry(a,b,c) → x
            self.add_clause([-a, -b, x])  # a ∧ b → x
            self.add_clause([-a, -c, x])  # a ∧ c → x
            self.add_clause([-b, -c, x])  # b ∧ c → x
        return x

    def encode_atleast_sequential_counter(self, lhs: Sequence[Lit], rhs: int, polarity: Polarity = Polarity(True, True)) -> Lit:
        tbl: List[Lit] = [lit_true] + [lit_false] * rhs
        for i, x in enumerate(lhs):
            j_min = max(1, rhs - (len(lhs) - i) + 1)
            for j in range(rhs, j_min - 1, -1):
                tbl[j] = self.encode_disj(tbl[j], self.encode_conj(tbl[j-1], x, polarity=polarity), polarity=polarity)
        return tbl[rhs]

    def encode_atleast_parallel_counter(self, lhs: Sequence[Lit], rhs: int, polarity: Polarity = Polarity(True, True)) -> Lit:
        def encode_rhs(n: int) -> Sequence[Lit]:
            ret = []
            while n != 0:
                ret.append(lit_true if n % 2 == 1 else lit_false)
                n //= 2
            return ret
        rhs_bits = encode_rhs(rhs)
        overflow_bits: List[Lit] = []

        def f(lits) -> List[Lit]:
            if len(lits) == 0:
                return []
            n = len(lits) // 2
            bin1 = f(lits[0:n])
            bin2 = f(lits[n:2*n])
            assert len(bin1) == len(bin2)
            bin3 = []
            c = lits[2*n] if len(lits) % 2 == 1 else lit_false
            for i in range(min(len(bin1), len(rhs_bits))):
                bin3.append(self.encode_fa_sum(bin1[i], bin2[i], c))
                c = self.encode_fa_carry(bin1[i], bin2[i], c)
            if len(bin3) == len(rhs_bits):
                overflow_bits.append(c)
            else:
                bin3.append(c)
            return bin3

        return self.encode_disj(self.encode_bvuge(f(lhs), rhs_bits, polarity=polarity), *overflow_bits, polarity=polarity)

    def encode_atleast_totalizer(self, lhs: Sequence[Lit], rhs: int) -> Lit:
        def encode_sum(lhs: Sequence[Lit]) -> Sequence[Lit]:
            if len(lhs) <= 1:
                return lhs
            else:
                n = len(lhs)
                xs1 = encode_sum(lhs[:n // 2])
                xs2 = encode_sum(lhs[n // 2:])
                rs = self.new_vars(n)
                for sigma in range(n+1):
                    # a + b = sigma, 0 <= a <= n1, 0 <= b <= n2
                    for a in range(max(0, sigma - len(xs2)), min(len(xs1), sigma) + 1):
                        b = sigma - a
                        # card(lits1) >= a ∧ card(lits2) >= b → card(lits) >= sigma
                        # ¬(card(lits1) >= a) ∨ ¬(card(lits2) >= b) ∨ card(lits) >= sigma
                        if sigma != 0:
                            self.add_clause(
                                ([- xs1[a - 1]] if a > 0 else []) + \
                                ([- xs2[b - 1]] if b > 0 else []) + \
                                [rs[sigma - 1]])
                        # card(lits) > sigma → (card(lits1) > a ∨ card(lits2) > b)
                        # card(lits) >= sigma+1 → (card(lits1) >= a+1 ∨ card(lits2) >= b+1)
                        # card(lits1) >= a+1 ∨ card(lits2) >= b+1 ∨ ¬(card(lits) >= sigma+1)
                        if sigma + 1 != n + 1:
                            self.add_clause(
                                ([xs1[a + 1 - 1]] if a + 1 < len(xs1) + 1 else []) + \
                                ([xs2[b + 1 - 1]] if b + 1 < len(xs2) + 1 else []) + \
                                [- rs[sigma + 1 - 1]])
                return rs
        if rhs <= 0:
            return lit_true
        elif len(lhs) < rhs:
            return lit_false
        else:
            lits = encode_sum(sorted(lhs))
            for i in range(len(lits)-1):
                self.add_clause([-lits[i+1], lits[i]])  # l2→l1 or equivalently ¬l1→¬l2
            return lits[rhs - 1]

    def encode_atleast(self, lhs: List[Lit], rhs: int, polarity: Polarity = Polarity(True, True)) -> Lit:
        if self._counter == "sequential":
            return self.encode_atleast_sequential_counter(lhs, rhs, polarity)
        elif self._counter == "parallel":
            return self.encode_atleast_parallel_counter(lhs, rhs, polarity)
        elif self._counter == "totalizer":
            return self.encode_atleast_totalizer(lhs, rhs)
        else:
            raise RuntimeError(f"unknown counter: \"{self._counter}\"")

    def encode_pb_atleast(self, constr: PBAtLeast, polarity: Polarity = Polarity(True, True)) -> Lit:
        constr = constr.normalize()
        if self._lb(constr.lhs) >= constr.rhs:
            return lit_true
        if self._ub(constr.lhs) < constr.rhs:
            return lit_false
        elif self._cnf:
            assert all(c == 1 for (c, _) in constr.lhs)
            return self.encode_atleast([x for (_, x) in constr.lhs], constr.rhs, polarity=polarity)
        else:
            y = self.new_var()
            if polarity.pos:
                # y → lhs ≥ rhs
                self.add_pb_atleast_soft(y, constr)
            if polarity.neg:
                # (lhs ≥ rhs → y) ⇔ (¬y → ¬(lhs ≥ rhs))
                self.add_pb_atleast_soft(-y, constr.negate())
            return y

    def _wbo_hint(self) -> str:
        num_equal = sum(1 for _w, constr in self.constrs if isinstance(constr, PBExactly))
        num_soft = sum(1 for w, _constr in self.constrs if w is not None)
        mincost = min((w for w, _constr in self.constrs if w is not None), default=0)
        maxcost = max((w for w, _constr in self.constrs if w is not None), default=0)
        sumcost = sum(w for w, _constr in self.constrs if w is not None)
        intsize = max(
            (
                1 + math.floor(math.log2(x)) if x > 0 else 0
                for x in [sumcost] + [sum(abs(c) for c, _lit in constr.lhs) + abs(constr.rhs) for _, constr in self.constrs]
            ),
            default=0,
        )
        return f"* #variable= {self.nvars} #constraint= {len(self.constrs)} #equal= {num_equal} intsize= {intsize} #soft= {num_soft} mincost= {mincost} maxcost= {maxcost} sumcost= {sumcost}\n"

    def write_to_file(self, filename: Union[str, Path]) -> None:
        with open(filename, "w", encoding="us-ascii") as f:
            if self._cnf:
                f.write(f"p cnf {self.nvars} {len(self.constrs)}\n")
                for w, constr in self.constrs:
                    assert w is None
                    self._write_clause(constr, f)
                    f.write('\n')
            else:
                f.write(self._wbo_hint())
                for w, constr in self.constrs:
                    f.write(f"{constr.to_pos_literals()} ;\n")

    def _write_clause(self, constr, f) -> None:
        assert isinstance(constr, PBAtLeast)
        assert all(c == 1 for c, v in constr.lhs) and constr.rhs == 1
        for _, v in constr.lhs:
            f.write(str(v))
            f.write(' ')
        f.write('0')

    def write_to_file_opt(self, filename: Union[str, Path]) -> None:
        with open(filename, "w", encoding="us-ascii") as f:
            top = sum(w for w, _ in self.constrs if w is not None) + 1
            if self._cnf:
                f.write(f"p wcnf {self.nvars} {len(self.constrs)} {top}\n")
                for w, constr in self.constrs:
                    if w is None:
                        f.write(str(top))
                    else:
                        f.write(str(w))
                    f.write(' ')
                    self._write_clause(constr, f)
                    f.write('\n')
            else:
                f.write(self._wbo_hint())
                f.write(f"soft: {top} ;\n")
                for w, constr in self.constrs:
                    if w is not None:
                        f.write(f"[{str(w)}] ")
                    f.write(f"{constr.to_pos_literals()} ;\n")


class BNNEncoder(Encoder):
    def encode_binarize_pixel(self, mu: float, sigma: float, gamma: float, beta: float, x: Sequence[Lit]) -> Lit:
        def byte_bits(x):
            return [lit_true if ((x >> i) & 1) > 0 else lit_false for i in range(8)]

        def bits_to_pbsum(x):
            return [(2**i, b) for (i, b) in enumerate(x)]

        # γ((x/255 - μ) / σ) + β >= 0
        # (γ/σ) (x/255 - μ) ≥ - β
        C_frac = 255 * (- beta * sigma / gamma + mu)
        if gamma >= 0:
            # x ≥ ⌈255 (- βσ/γ + μ)⌉ = C
            C = int(math.ceil(C_frac))
            if self._cnf:
                return self.encode_bvuge(x, byte_bits(C))
            else:
                return self.encode_pb_atleast(PBAtLeast(bits_to_pbsum(x), C))
        else:
            # x ≤ ⌊255 (- βσ/γ + μ)⌋ = C
            C = int(math.floor(C_frac))
            if self._cnf:
                return self.encode_bvuge(byte_bits(C), x)
            else:
                # -x ≥ -C
                return self.encode_pb_atleast(PBAtLeast([(-c, v) for (c, v) in bits_to_pbsum(x)], -C))

    def encode_binarize_image(self, input_bn, image: Sequence[Sequence[Lit]]) -> Sequence[Lit]:
        mu = input_bn.avg_mean
        sigma = np.sqrt(input_bn.avg_var + input_bn.eps)
        gamma = input_bn.gamma.array
        beta = input_bn.beta.array
        return [self.encode_binarize_pixel(mu[i], sigma[i], gamma[i], beta[i], x) for i, x in enumerate(image)]

    def encode_block_1(self, row, b: float, mu: float, sigma: float, gamma: float, beta: float, xs: Sequence[Lit]) -> Lit:
        # assert all(int(x) == 1 or int(x) == -1 for x in row.reshape(-1))
        # γ((Σ_i row[i]*(2*xs[i]-1) + b) - μ)/σ + β ≥ 0
        # ((Σ_i row[i]*(2*xs[i]-1) + b) - μ)*(γ/σ) ≥ -β
        C_frac = (- beta * sigma / gamma + mu - b + sum(row)) / 2
        if gamma >= 0:
            # Σ_i row[i]*xs[i] ≥ ⌈(-βσ/γ + μ - b + Σ_i row[i]) / 2⌉ = C
            C = int(math.ceil(C_frac))
            return self.encode_pb_atleast(PBAtLeast([(int(row[i]), x) for (i, x) in enumerate(xs)], C))
        else:
            # Σ_i row[i]*xs[i] ≤ ⌊(-βσ/γ + μ - b + Σ_i row[i]) / 2⌋ = C
            C = int(math.floor(C_frac))
            # Σ_i -row[i]*xs[i] ≥ -C
            return self.encode_pb_atleast(PBAtLeast([(-int(row[i]), x) for (i, x) in enumerate(xs)], -C))

    def encode_block(self, block: bnn.Block, xs: Sequence[Lit]) -> List[Lit]:
        n_output, n_input = block.lin.W.shape
        assert len(xs) == n_input
        W = block.lin.W.array.astype(np.int32)
        b = block.lin.b.array
        mu = block.bn.avg_mean
        sigma = np.sqrt(block.bn.avg_var + block.bn.eps)
        gamma = block.bn.gamma.array
        beta = block.bn.beta.array
        assert all(int(x) == 1 or int(x) == -1 for x in W.astype(np.int32).reshape(-1))
        return [self.encode_block_1(W[i].astype(np.int32), b[i], mu[i], sigma[i], gamma[i], beta[i], xs) for i in range(n_output)]

    def encode_output(self, lin: bnn.BinLinear, xs: Sequence[Lit], output: Sequence[Lit]) -> None:
        m = len(xs)
        n = len(output)
        if self._cnf:
            self.add_clause(output)
            for i in range(len(output)):
                o1 = output[i]
                for o2 in output[i+1:]:
                    self.add_clause([-o1, -o2])
        else:
            self.add_pb_exactly(PBExactly([(1, x) for x in output], 1))

        W = lin.W.array.astype(np.int32)
        b = lin.b.array
        assert W.shape == (n, m)
        assert b.shape == (n, )
        #   (Σ_j W[i,j] (2xs[j]-1)) + b[i]
        # = (Σ_j 2 W[i,j] xs[j]) - (Σ_j W[i,j]) + b[i]
        logits = [([(2 * int(W[i, j]), xs[j]) for j in range(m)], - sum(W[i, j] for j in range(m)) + b[i]) for i in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # logits[i] >= logits[j]
                constr = PBAtLeast(logits[i][0] + [(-c, v) for (c, v) in logits[j][0]], int(math.ceil(logits[j][1] - logits[i][1])))
                if self._cnf:
                    constr = constr.normalize()
                    assert all(c == 1 for c, v in constr.lhs)
                    x = self.encode_atleast([v for _, v in constr.lhs], constr.rhs, Polarity(True, False))
                    self.add_clause([-output[i], x])
                else:
                    self.add_pb_atleast_soft(output[i], constr)

    def encode(self, model: bnn.BNN, image: Sequence[Sequence[Lit]], output: Sequence[Lit]) -> None:
        h = self.encode_binarize_image(model.input_bn, image)
        self.encode_bin_input(model, h, output)

    def encode_bin_input(self, model: bnn.BNN, image: Sequence[Lit], output: Sequence[Lit]) -> None:
        h = image
        for block in model.blocks:
            h = self.encode_block(block, h)
        self.encode_output(model.output_lin, h, output)

    def add_norm_cost(self, norm: str, mod: Sequence[Tuple[Lit, Optional[int]]]) -> None:
        if norm == '0':
            for lit, w in mod:
                self.add_clause([lit], None if w is None else 1)
        elif norm == '1':
            for lit, w in mod:
                self.add_clause([lit], None if w is None else abs(w))
        elif norm == '2':
            for lit, w in mod:
                self.add_clause([lit], None if w is None else w*w)
        elif norm == 'inf':
            d: Dict[int, List[Lit]] = {}
            for lit, w in mod:
                if w is None:
                    self.add_clause([lit])
                else:
                    w = abs(w)
                    if w not in d:
                        d[w] = []
                    d[w].append(lit)
            c_prev = 0
            relax_prev = None
            for w in sorted(d.keys()):
                relax = self.new_var()
                self.add_clause([-relax], cost=w - c_prev)
                if relax_prev is not None:
                    self.add_clause([-relax, relax_prev])  # relax → relax_prev
                for lit in d[w]:
                    self.add_clause([relax, lit])  # ¬lit → relax
                c_prev = w
                relax_prev = relax
        else:
            raise RuntimeError("unknown norm: " + str(norm))
