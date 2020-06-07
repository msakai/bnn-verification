import functools
import operator
import chainer
from chainer import Chain, ChainList, functions as F, Link, links as L
from chainer.training import extensions
import numpy as np
from typing import List


def hard_tanh(x):
    return F.clip(x, -1.0, 1.0)


def hard_sigmoid(x):
    return F.clip((x + 1) / 2, 0.0, 1.0)


def bin(x, stochastic=False):
    if stochastic:
        p = hard_sigmoid(x)
        y_hard = chainer.distributions.Bernoulli(p).sample().array * 2 - 1
        y_hard = chainer.Variable(y_hard.astype(np.float32))
    else:
        y_hard = F.sign(x)

    if chainer.config.enable_backprop:
        y_soft = hard_tanh(x)
        return (y_hard - y_soft.array) + y_soft
    else:
        return y_hard


class BinLinear(Link):
    def __init__(self, in_size, out_size=None, nobias=False,
                 initialW=None, initial_bias=None):
        super().__init__()

        if out_size is None:
            in_size, out_size = None, in_size
        self.out_size = out_size

        with self.init_scope():
            W_initializer = chainer.initializers._get_initializer(initialW)
            self.W = chainer.Parameter(W_initializer)
            if in_size is not None:
                self._initialize_params(in_size)

            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = chainer.initializers._get_initializer(initial_bias)
                self.b = chainer.Parameter(bias_initializer, out_size)

    def _initialize_params(self, in_size):
        self.W.initialize((self.out_size, in_size))

    def forward(self, x, n_batch_axes=1):
        if self.W.array is None:
            in_size = functools.reduce(operator.mul, x.shape[1:], 1)
            self._initialize_params(in_size)
        W = bin(self.W)
        return F.linear(x, W, self.b, n_batch_axes=n_batch_axes)
      
    def clip_weights(self):
        with chainer.cuda.get_device(self.W.array):
            xp = chainer.cuda.get_array_module(self.W.array)
            self.W.array = xp.clip(self.W.array, -1.0, 1.0)
            
    def binarize_weights(self):
        with chainer.cuda.get_device(self.W.array):
            xp = chainer.cuda.get_array_module(self.W.array)
            self.W.array = xp.sign(self.W.array)

class Block(Chain):
    def __init__(self, n_output: int, stochastic_activation: bool = False) -> None:
        super().__init__()
        self.stochastic_activation = stochastic_activation
        with self.init_scope():
            self.lin = BinLinear(None, n_output)
            self.bn = L.BatchNormalization(n_output)

    def __call__(self, x):
      return bin(self.bn(self.lin(x)), self.stochastic_activation and chainer.config.train)

    def clip_weights(self):
        self.lin.clip_weights()

    def binarize_weights(self):
        self.lin.binarize_weights()


class BNN(Chain):
    def __init__(self, neurons: List[int], stochastic_activation=False) -> None:
        super().__init__()
        with self.init_scope():
            self.input_bn = L.BatchNormalization(neurons[0])
            self.blocks = ChainList(*[Block(n, stochastic_activation=stochastic_activation) for n in neurons[1:-1]])
            self.output_lin = BinLinear(None, neurons[-1])

    def __call__(self, x):
        h = bin(self.input_bn(x))
        for block in self.blocks:
            h = block(h)
        return self.output_lin(h)

    def forward_with_intermediate_results(self, x):
        result = []
        h = bin(self.input_bn(x))
        result.append(h)
        for block in self.blocks:
            h = block(h)
            result.append(h)
        result.append(self.output_lin(h))
        return result

    def saliency_map(self, x, y):
        assert x.shape == (28*28,)
        x = chainer.Variable(x)
        logits = self(x.reshape(1, -1))
        return chainer.grad([logits[0, y]], [x])[0].array

    def clip_weights(self):
        for block in self.blocks:
            block.clip_weights()

    def binarize_weights(self):
        for block in self.blocks:
            block.binarize_weights()
