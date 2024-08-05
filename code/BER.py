# -*- coding: utf-8 -*-
# @Time    : 2023/11/18 11:21
# @Author  :Fivem
# @File    : BER.py
# @Software: PyCharm
# @last modified:2023/11/18 11:21

import numpy as np


def BER(original_watermark, extracted_watermark):
    print(f"原始水印长度为{len(original_watermark)}\n提取水印长度为{len(extracted_watermark)}")
    if len(extracted_watermark) != len(original_watermark):
        exit('Input watermark must be the same size!')
        # if len(extracted_watermark) < len(original_watermark):
        #     print("提取水印长度小于原始水印长度，将给提取的水印补0")
        #     extracted_watermark = np.hstack(
        #         (extracted_watermark, [0] * (len(original_watermark) - len(extracted_watermark))))
        # else:
        #     print("原始水印长度小于提取水印长度，将给原始的水印补0")
        #     original_watermark = np.hstack(
        #         (original_watermark, [0] * (len(extracted_watermark) - len(original_watermark))))
    result = original_watermark ^ extracted_watermark
    p = np.sum(result) / np.size(original_watermark)
    # print(f"BER结果为{p}")
    return f'{p}'


if __name__ == "__main__":
    original_watermark = np.array([[0, 1], [0, 1]])
    extract_watermark = np.array([[0, 1], [0, 0]])
    print(BER(original_watermark, extract_watermark))
