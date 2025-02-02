import chainer
import numpy as np

import bnn


def convert(input_npz, output_npz):
    neurons = [28 * 28, 200, 100, 100, 100, 10]
    model = bnn.BNN(neurons, stochastic_activation=True)

    chainer.serializers.load_npz(input_npz, model)

    input_bn = model.input_bn
    block0 = model.blocks[0]

    input_bn.gamma.array = input_bn.gamma.array.reshape(28,28).T.reshape(-1)
    input_bn.beta.array = input_bn.beta.array.reshape(28,28).T.reshape(-1)
    input_bn.avg_mean = input_bn.avg_mean.reshape(28,28).T.reshape(-1)
    input_bn.avg_var = input_bn.avg_var.reshape(28,28).T.reshape(-1)
    block0.lin.W.array = np.swapaxes(block0.lin.W.array.reshape(-1,28,28), 1, 2).reshape(-1,28*28)

    chainer.serializers.save_npz(output_npz, model)


convert("models/mnist_back_image.npz", "models/mnist_back_image_reordered.npz")
convert("models/mnist_rot.npz", "models/mnist_rot_reordered.npz")
