# -*-  coding=utf-8 -*-
# @Time : 2022/3/24 18:10
# @Author : Scotty1373
# @File : utils_tools_dataset.py
# @Software : PyCharm
from PIL import Image, ImageDraw
import numpy as np


def get_image(file_path, *, dtype='float32'):
    fp = Image.open(file_path)
    img = np.array(fp, dtype)

    if img.ndim != 3:
        # HW --> CHW
        img = img[None, ...]
    else:
        # HWC --> CHW
        img = img.transpose((2, 0, 1))

    return img


def bbox_resize():
    pass
