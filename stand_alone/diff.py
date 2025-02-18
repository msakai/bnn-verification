import argparse

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("image1", type=str, help="input image 1 (*.png)")
parser.add_argument("image2", type=str, help="input image 2 (*.png)")
args = parser.parse_args()

x1 = np.array(Image.open(args.image1)).reshape(-1).astype(np.int32)
x2 = np.array(Image.open(args.image2)).reshape(-1).astype(np.int32)
diff = x2 - x1

for norm in [0, 1, 2, np.inf]:
    z = np.linalg.norm(diff, ord=norm)
    print(f"{norm}-norm: {z}")
