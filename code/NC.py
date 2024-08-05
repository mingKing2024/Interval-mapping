# -*- coding: utf-8 -*-
# @Time    : 2023/11/8 15:40
# @Author  :Fivem
# @File    : NC.py
# @Software: PyCharm
# @last modified:2023/11/8 15:40

import numpy as np
from PIL import Image


def image_to_array(path):
    image = Image.open(path)  # 替换为你的PNG格式二值图像文件路径
    # 转换为NumPy数组
    image_array = np.array(image).astype(int)
    return image_array


def NC(original_watermark, extract_watermark):
    """
    calculate normalized correlation(NC)
    :param original_watermark:
    :param extract_watermark:
    :return:
    """
    if original_watermark.shape != extract_watermark.shape:
        exit('Input watermark must be the same size!')
    elif ~np.all((original_watermark == 0) | (original_watermark == 1)) | ~np.all(
            (extract_watermark == 0) | (extract_watermark == 1)):
        exit('The input must be a binary image logical value image!')

    result = np.sum(original_watermark * extract_watermark) / (
            np.sqrt(np.sum(original_watermark ** 2)) * np.sqrt(np.sum(extract_watermark ** 2)))
    return result


if __name__ == "__main__":
    original_watermark = np.array([[0, 1], [0, 1]])
    extract_watermark = np.array([[0, 1], [0, 0]])
    print(NC(original_watermark, extract_watermark))
