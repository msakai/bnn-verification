# Stand-alone version

This directory contains stand-alone version that takes a trained PyTorch model and an input image.

- `diff.py` is a script for computing differences of two images
- `predict.py` is a script for performing predition
- `solve_gurobi.py` is a script for solving the problems

If your model is trained using column-major feature order, you need to pass `--column-major` option to `solve_gurobi.py` and `predict.py`.
You need it for [mnist\_back\_image.pt](../models/mnist_back_image.pt) and [mnist\_rot.pt](../models/mnist_rot.pt), but not for [mnist.pt](../models/mnist.pt), [mnist\_back\_image\_reordered.pt](../models/mnist_back_image_reordered.pt), and [mnist\_rot\_reordered.pt](../models/mnist_rot_reordered.pt).
