import mindspore
from mindspore import Tensor
import os
from PIL import Image
import numpy as np

def pil_loader(path, gray=False):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if gray:
            return img.convert('L')
        else:
            return img.convert('RGB')

def create_dataset(args):
    image = pil_loader('dataset.jpg')
    normalize = mindspore.dataset.vision.py_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                 std=[0.5, 0.5, 0.5])

    image = mindspore.dataset.vision.py_transforms.Resize((args.img_size_max, args.img_size_max))(image)
    image = mindspore.dataset.vision.py_transforms.ToTensor()(image)
    image = normalize(image)
    return Tensor(image)
