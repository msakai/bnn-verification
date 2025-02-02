import os
import zipfile

import numpy as np

import chainer
from chainer.dataset import download
from chainer.datasets.mnist import preprocess_mnist


def get_mnist_back_image(withlabel=True, ndim=1, scale=1., dtype=None,
                         label_dtype=np.int32, rgb_format=False, reorder=False):
    """Gets the MNIST-back-image dataset.

    `MNIST + background images <http://web.archive.org/web/20180831072509/http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/MnistVariations>`_ is a
    variant of MNIST where a patch from a black and white image was used as
    the background for the digit image.

    See :func:`chainer.datasets.get_mnist` for the details of arguments and
    return values.
    """
    return _get_mnist_variation(
        "mnist_background_images",
        "mnist_background_images_train",
        "mnist_background_images_test",
        withlabel=withlabel, ndim=ndim, scale=scale, dtype=dtype,
        label_dtype=label_dtype, rgb_format=rgb_format,
        reorder=reorder,
    )


def get_mnist_rot(withlabel=True, ndim=1, scale=1., dtype=None,
                  label_dtype=np.int32, rgb_format=False, reorder=False):
    """Gets the rotated MNIST digits dataset.

    `Rotated MNIST digits <http://web.archive.org/web/20180831072509/http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/MnistVariations>`_ is a
    variant of MNIST where the digits were rotated by an angle generated
    uniformly between 0 and :math:`2 \\pi` radians.

    See :func:`chainer.datasets.get_mnist` for the details of arguments
    and return values.
    """
    return _get_mnist_variation(
        "mnist_rotation_new",
        "mnist_all_rotation_normalized_float_train_valid",
        "mnist_all_rotation_normalized_float_test",
        withlabel=withlabel, ndim=ndim, scale=scale, dtype=dtype,
        label_dtype=label_dtype, rgb_format=rgb_format,
        reorder=reorder,
    )


def _get_mnist_variation(
        name, train_filename_base, test_filename_base,
        withlabel=True, ndim=1, scale=1., dtype=None,
        label_dtype=np.int32, rgb_format=False, reorder=False):
    url = "http://www.iro.umontreal.ca/~lisa/icml2007data/" + name + ".zip"
    root = download.get_dataset_directory('msakai/mnist_variation/')

    path = os.path.join(root, train_filename_base + ".npz")
    train_raw = download.cache_or_load_file(
        path,
        lambda path: _make_npz(path, url, train_filename_base + ".amat"),
        np.load)

    path = os.path.join(root, test_filename_base + ".npz")
    test_raw = download.cache_or_load_file(
        path,
        lambda path: _make_npz(path, url, test_filename_base + ".amat"),
        np.load)

    dtype = chainer.get_dtype(dtype)
    train = preprocess_mnist(train_raw, withlabel, ndim, scale, dtype,
                             label_dtype, rgb_format)
    test = preprocess_mnist(test_raw, withlabel, ndim, scale, dtype,
                            label_dtype, rgb_format)
    if reorder:
        def f(x):
            x = x.reshape(-1, 28, 28)
            x = np.swapaxes(x, 1, 2)
            x = x.reshape(-1, 28*28)
            return x
        assert len(train._datasets) == 2
        assert len(test._datasets) == 2
        train._datasets = (f(train._datasets[0]), train._datasets[1])
        test._datasets = (f(test._datasets[0]), test._datasets[1])

    return train, test


def _make_npz(path, url, fname):
    zip_path = download.cached_download(url)
    with zipfile.ZipFile(zip_path) as z:
        with z.open(fname) as f:
            mat = np.loadtxt(f, dtype=np.float64)
            x = np.round(mat[:, :-1] * 255).astype(np.uint8)
            y = mat[:, -1].astype(np.uint8)
    np.savez_compressed(path, x=x, y=y)
    return {'x': x, 'y': y}


def get_dataset(name: str, *args, **kwargs):
    if name == "mnist":
        return chainer.datasets.get_mnist(*args, **kwargs)
    elif name == "mnist_back_image":
        return get_mnist_back_image(*args, **kwargs)
    elif name == "mnist_rot":
        return get_mnist_rot(*args, **kwargs)
    elif name == "mnist_back_image_reordered":
        return get_mnist_back_image(*args, **kwargs, reorder=True)
    elif name == "mnist_rot_reordered":
        return get_mnist_rot(*args, **kwargs, reorder=True)
    else:
        raise RuntimeError("unknown dataset: " + name)
