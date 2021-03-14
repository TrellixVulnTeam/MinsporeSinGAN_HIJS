import mindspore
from mindspore import Tensor
import os
import cv2

def create_dataset():
    image = cv2.imread('dataset.jpg')
    image = Tensor(image, mindspore.int8)
    return image

