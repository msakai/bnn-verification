import torch
import numpy as np

from bnn_pytorch import BNN


def load_chainer_weight(model, fname):
    state_dict = model.state_dict()

    weights = np.load(fname)

    mapping = {
        'input_block.0.weight': 'input_bn/gamma',
        'input_block.0.bias': 'input_bn/beta',
        'input_block.0.running_mean': 'input_bn/avg_mean',
        'input_block.0.running_var': 'input_bn/avg_var',
        'input_block.0.num_batches_tracked': 'input_bn/N',

        'hidden_layers.0.0.weight': 'blocks/0/lin/W',
        'hidden_layers.0.0.bias': 'blocks/0/lin/b',
        'hidden_layers.0.1.weight': 'blocks/0/bn/gamma',
        'hidden_layers.0.1.bias': 'blocks/0/bn/beta',
        'hidden_layers.0.1.running_mean': 'blocks/0/bn/avg_mean',
        'hidden_layers.0.1.running_var': 'blocks/0/bn/avg_var',
        'hidden_layers.0.1.num_batches_tracked': 'blocks/0/bn/N',

        'hidden_layers.1.0.weight': 'blocks/1/lin/W',
        'hidden_layers.1.0.bias': 'blocks/1/lin/b',
        'hidden_layers.1.1.weight': 'blocks/1/bn/gamma',
        'hidden_layers.1.1.bias': 'blocks/1/bn/beta',
        'hidden_layers.1.1.running_mean': 'blocks/1/bn/avg_mean',
        'hidden_layers.1.1.running_var': 'blocks/1/bn/avg_var',
        'hidden_layers.1.1.num_batches_tracked': 'blocks/1/bn/N',

        'hidden_layers.2.0.weight': 'blocks/2/lin/W',
        'hidden_layers.2.0.bias': 'blocks/2/lin/b',
        'hidden_layers.2.1.weight': 'blocks/2/bn/gamma',
        'hidden_layers.2.1.bias': 'blocks/2/bn/beta',
        'hidden_layers.2.1.running_mean': 'blocks/2/bn/avg_mean',
        'hidden_layers.2.1.running_var': 'blocks/2/bn/avg_var',
        'hidden_layers.2.1.num_batches_tracked': 'blocks/2/bn/N',

        'hidden_layers.3.0.weight': 'blocks/3/lin/W',
        'hidden_layers.3.0.bias': 'blocks/3/lin/b',
        'hidden_layers.3.1.weight': 'blocks/3/bn/gamma',
        'hidden_layers.3.1.bias': 'blocks/3/bn/beta',
        'hidden_layers.3.1.running_mean': 'blocks/3/bn/avg_mean',
        'hidden_layers.3.1.running_var': 'blocks/3/bn/avg_var',
        'hidden_layers.3.1.num_batches_tracked': 'blocks/3/bn/N',

        'output_layer.weight': 'output_lin/W',
        'output_layer.bias': 'output_lin/b'
    }

    assert set(state_dict.keys()) == set(mapping.keys())
    assert set(weights.keys()) == set(v for v in mapping.values())
    for param_pytorch, param_chainer in mapping.items():
        assert state_dict[param_pytorch].shape == weights[param_chainer].shape

    state_dict = {
        param_pytorch: torch.tensor(weights[param_chainer])
        for param_pytorch, param_chainer in mapping.items()
    }
    model.load_state_dict(state_dict)


model = BNN(hidden_layers=[200, 100, 100, 100])
model.eval()

load_chainer_weight(model, "models/mnist.npz")
torch.save(model.state_dict(), "models/mnist.pt")

load_chainer_weight(model, "models/mnist_rot.npz")
torch.save(model.state_dict(), "models/mnist_rot.pt")

load_chainer_weight(model, "models/mnist_back_image.npz")
torch.save(model.state_dict(), "models/mnist_back_image.pt")

load_chainer_weight(model, "models/mnist_rot_reordered.npz")
torch.save(model.state_dict(), "models/mnist_rot_reordered.pt")

load_chainer_weight(model, "models/mnist_back_image_reordered.npz")
torch.save(model.state_dict(), "models/mnist_back_image_reordered.pt")
