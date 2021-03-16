# import mindspore
# from mindspore import Tensor
# import os
# import numpy as np
# import cv2

# image = cv2.imread('dataset.jpg')
# image = Tensor(image, mindspore.int8)

# print(image.shape)

# pad = mindspore.ops.Pad(((0, 0), (0, 0), (5, 5), (5, 5)))


# batch_size = 1


# z_fix_list = [pad(
#     Tensor(
#         np.random.randn(1, 3, 25, 25), mindspore.float32
#     )
# )]

# print(z_fix_list[0].shape)

# 闭包=====只有一个方法的对象，他捕获的数据，就是这个对象的属性

