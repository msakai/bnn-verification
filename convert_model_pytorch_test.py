import chainer
import torch
import numpy as np

import bnn as bnn_chainer
import bnn_pytorch


def check(chainer_weights, torch_weights):
    model_chainer = bnn_chainer.BNN([28 * 28, 200, 100, 100, 100, 10], stochastic_activation=True)
    chainer.serializers.load_npz(chainer_weights, model_chainer)

    model_pytorch = bnn_pytorch.BNN(hidden_layers=[200, 100, 100, 100])
    model_pytorch.load_state_dict(torch.load(torch_weights, map_location="cpu"))
    model_pytorch.eval()

    image_scaled = np.random.rand(1, 28 * 28).astype(np.float32)

    with chainer.using_config("train", False), chainer.using_config("enable_backprop", False):
        chainer_result = model_chainer(image_scaled).array
        print("Chainer:")
        print(chainer_result)

    with torch.inference_mode():
        pytorch_result = model_pytorch(torch.tensor(image_scaled, requires_grad=False)).numpy()
    print("PyTorch:")
    print(pytorch_result)

    print(f"All equals?: {np.all(chainer_result == pytorch_result)}")


print("models/mnist")
check("models/mnist.npz", "models/mnist.pt")

print("")

print("models/mnist_rot")
check("models/mnist_rot.npz", "models/mnist_rot.pt")

print("")

print("models/mnist_back_image")
check("models/mnist_back_image.npz", "models/mnist_back_image.pt")

print("")

print("models/mnist_rot_reordered")
check("models/mnist_rot_reordered.npz", "models/mnist_rot_reordered.pt")

print("")

print("models/mnist_back_image")
check("models/mnist_back_image_reordered.npz", "models/mnist_back_image_reordered.pt")
