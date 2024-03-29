import numpy as np
import PIL


def to_image(dataset_name: str, data):
    """
    This function convert feature vector into a image

    Note that the sequence of pixels in the feature vector differs depending on datasets.
    This difference should have been resolved when creating the data set.
    But we didn't noticed that at that time.
    """

    data = data.reshape(28, 28)
    if dataset_name == "mnist":
        pass
    elif dataset_name == "mnist_back_image":
        data = np.flipud(data)
        data = np.rot90(data, 3)
    elif dataset_name == "mnist_rot":
        data = np.flipud(data)
        data = np.rot90(data, 3)
    else:
        raise RuntimeError("unknown dataset: " + dataset_name)    
    return PIL.Image.fromarray(data)
