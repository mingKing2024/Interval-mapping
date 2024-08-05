# -*- coding: utf-8 -*-
# @Time    : 2024/2/14 22:34
# @Author  :Fivem
# @File    : extract.py
# @Software: PyCharm
# @last modified:2024/2/14 22:34


import math
import os
import random
from collections import Counter
from decimal import Decimal

import geopandas as gpd
import numpy as np
from PIL import Image
from BER import BER

from NC import NC, image_to_array
from get_coor import get_coor_nested, get_coor_array
from select_file import select_file
from to_geodataframe import to_geodataframe


def watermark_extract(coor, coor_l, R, n):
    """
    对坐标嵌入水印
    :param coor: 变换后的坐标
    :param coor_l: 区域左下角顶点的坐标
    :return:返回嵌入水印的坐标
    """
    x, y = coor
    xl, yl = coor_l
    x, y = x, y
    xl, yl = xl, yl
    w = (x - xl) // (R / 2 ** n)
    original_x = xl + 2 ** n * (x - xl) - w * R
    original_y = y
    return np.vstack([original_x, original_y]), int(w)


def coor_process(coor, W, dis, argument):
    """
    对坐标进行处理
    :param coor: 需要处理的坐标
    :return: 嵌入水印的坐标
    """
    # print(coor)
    n, R, side_length, ratio = argument.values()
    random.seed(dis)

    index = random.randint(0, len(W) - 1)

    coor_l = coor // R * R
    original_coor, w = watermark_extract(coor, coor_l, R, n)
    W[index].append(w)
    # print(dis,index,w)
    return original_coor, W


def coor_group_process(coor_group, W, argument):
    """
    对坐标组进行处理
    :param coor_group:需要处理的坐标组
    :return: 返回嵌入水印的坐标组
    """
    n, R, side_length, ratio = argument.values()

    coor_group = np.array([[Decimal(str(x)) for x in row] for row in coor_group])
    ratio = Decimal(str(ratio))
    coor_group = coor_group * ratio

    original_coor_group = np.array([[], []])
    for i in range(coor_group.shape[1]):
        coor = coor_group[:, i]
        if i == coor_group.shape[1] - 1:
            original_coor = coor[:, np.newaxis]
        else:
            dis = round(abs(coor[1] - coor_group[:, i + 1][1]), 9)

            original_coor, W = coor_process(coor, W, dis, argument)
        original_coor_group = np.concatenate((original_coor_group, original_coor), axis=1)

    return original_coor_group, W


def traversal_nested_coor_group(coor_nested, feature_type, W, argument):
    """
    对于多线、多面等情况，执行此函数
    :param coor_nested: 所有要素组成的嵌套坐标数组
    :param feature_type: 要素的类型
    :return: 返回该要素更新后的嵌套坐标组
    """
    processed_x_nested = []
    processed_y_nested = []
    # 遍历要素中的每个坐标组
    for feature_index in range(coor_nested.shape[1]):
        coor_group = np.vstack(coor_nested[:, feature_index])
        # 对坐标进行平移
        processed_coor_group, W = coor_group_process(coor_group, W, argument)
        # 如果要素为多面，则需要满足首位顶点的坐标相同
        if (feature_type == 'MultiPolygon'
                and np.size(processed_coor_group) not in [0, 2]
                and not np.array_equal(processed_coor_group[:, 0], processed_coor_group[:, -1])):
            processed_coor_group[:, -1] = processed_coor_group[:, 0]
        processed_x_nested.append(processed_coor_group[0, :])
        processed_y_nested.append(processed_coor_group[1, :])
    return np.array([processed_x_nested, processed_y_nested], dtype=object), W


def traversal_coor_group(coor_nested, shp_type, processed_shpfile, l, argument):
    """
    对所有要素进行遍历
    :param coor_nested: 所有要素组成的嵌套坐标数组
    :param shp_type: 每个要素类型组成的数组
    :param processed_shpfile: 处理后的shp文件
    :return: processed_shpfile
    """
    # ----------------定义局部变量----------------------
    W = [[] for _ in range(l)]

    # 遍历每个几何要素
    for feature_index in range(coor_nested.shape[1]):
        coor_group = np.vstack(coor_nested[:, feature_index])
        if isinstance(coor_nested[:, feature_index][0], list):
            continue
        feature_type = shp_type[feature_index]
        # 判断是否是多线、多面等的情况
        if isinstance(coor_group[0, 0], np.ndarray):
            processed_coor_group, W = traversal_nested_coor_group(coor_group, feature_type, W, argument)
        else:
            processed_coor_group, W = coor_group_process(coor_group, W, argument)
            # 如果要素为面，则需要满足首尾顶点的坐标相同
            if (feature_type == 'Polygon'
                    and np.size(processed_coor_group) not in [0, 2]
                    and not np.array_equal(processed_coor_group[:, 0], processed_coor_group[:, -1])):
                processed_coor_group[:, -1] = processed_coor_group[:, 0]
        # 将改变的要素坐标组更新到geodataframe
        processed_shpfile = to_geodataframe(processed_shpfile, feature_index, processed_coor_group,
                                            shp_type[feature_index])
    return processed_shpfile, W


def calculate_watermark_and_nc(watermarked_shpfile, l, argument):
    watermarked_coor_nested, feature_type = get_coor_nested(watermarked_shpfile)
    n, R, side_length, ratio = argument.values()

    # -------------------------水印提取--------------------------------
    # 遍历每个对象
    original_shpfile = watermarked_shpfile.copy()
    original_shpfile, W = traversal_coor_group(watermarked_coor_nested, feature_type, original_shpfile, l,
                                               argument)

    # 计算每个数组出现的次数
    t = 0
    empty_array_flag = False  # 标志变量，用于跟踪是否存在空数组
    for i in range(len(W)):
        # 使用Counter统计每个数字的出现次数
        counter = Counter(W[i])
        if counter:
            w = counter.most_common(1)[0][0]
            w = 0 if w < 0 else w  # 避免提取的水印有小数
            W[i] = [int(digit) for digit in format(w, f"0{n}b")]  # 转换n位二进制
        else:
            empty_array_flag = True
            t += 1
            W[i] = [0] * n
    # 在循环结束后检查标志变量并打印消息
    if empty_array_flag:
        print(f'存在{t}个空数组，补零')
    W = [item for w in W for item in w]

    W = W[:side_length ** 2 - 192]  # 截断多余的长度
    # print(W,len(W))
    # 将计算后的数组填充到原数组中
    watermark = np.full((25, 25), 1)
    detection_pattern = np.full((7, 7), 0)
    detection_pattern[1:6, 1:6] = np.full((5, 5), 1)
    detection_pattern[2:5, 2:5] = np.full((3, 3), 0)
    watermark[:7, :7] = watermark[:7, -7:] = watermark[-7:, :7] = detection_pattern
    watermark[0:8, 8: -8] = np.reshape(W[0:(side_length - 16) * 8], (8, side_length - 16))
    watermark[8:-8, :] = np.reshape(W[(side_length - 16) * 8:-(side_length - 8) * 8], (side_length - 16, side_length))
    watermark[-8:, 8:] = np.reshape(W[-(side_length - 8) * 8:], (8, side_length - 8))

    return original_shpfile, watermark


def extract(watermarked_shpfile_path, original_watermark_path, vr):
    # -------------------------预定义--------------------------------
    n = 4  # 嵌入强度
    tau = 10 ** (-6)  # 精度容差
    side_length = 25  # 水印图像边长
    l = math.ceil((side_length ** 2 - 192) / n)  # 水印长度
    R = Decimal('1e-7')
    ratio = 1
    # R = 1e-7

    # -------------------------数据读取--------------------------------
    watermarked_shpfile = gpd.read_file(watermarked_shpfile_path)

    # -------------------------数据预处理--------------------------------
    print('倍数', ratio)
    argument = {'n': n, 'R': R, 'side_length': side_length, 'ratio': ratio, }

    original_shpfile1, watermark1 = calculate_watermark_and_nc(watermarked_shpfile, l, argument)

    # 评估NC值
    original_watermark = image_to_array(original_watermark_path)
    nc1 = NC(original_watermark, watermark1)
    print(f'NC值为{nc1}')
    if abs(nc1 - 1) <= 1e-15:
        nc, original_shpfile, watermark = nc1, original_shpfile1, watermark1
    else:
        print('*' * 10)
        print('*' * 10)
        watermarked_coor_nested, feature_type = get_coor_nested(watermarked_shpfile)
        coor_array = get_coor_array(watermarked_coor_nested, feature_type)  # 将嵌套坐标数组合并成一个数组
        coor_mean = np.mean(coor_array, axis=1)
        ratio = np.mean(vr / coor_mean)
        print('倍数', ratio)
        argument = {'n': n, 'R': R, 'side_length': side_length, 'ratio': ratio, }
        original_shpfile2, watermark2 = calculate_watermark_and_nc(watermarked_shpfile, l, argument)
        # 评估NC值
        nc2 = NC(original_watermark, watermark2)
        print(f'NC值为{nc2}')
        nc, original_shpfile, watermark = (nc2, original_shpfile2, watermark2) if nc2 >= nc1 else (
            nc1, original_shpfile1, watermark1)
    ber = BER(original_watermark, watermark)
    print(f'NC值为{nc},BER值为{ber}')
    error = {'NC': nc, 'BER': ber}
    # -------------------------数据输出--------------------------------
    # 创建文件夹
    folder_name = '..\\extract'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if not os.path.exists(f'{folder_name}\\shpfile'):
        os.makedirs(f'{folder_name}\\shpfile')

    if not os.path.exists(f'{folder_name}\\watermark'):
        os.makedirs(f'{folder_name}\\watermark')

    output_shapefile_path = f'{folder_name}\\shpfile\\{os.path.basename(watermarked_shpfile_path)}'
    # 将GeoDataFrame保存为shp文件
    original_shpfile.to_file(output_shapefile_path)
    print("Shapefile创建完成，已保存为", output_shapefile_path)

    output_watermark_path = f'{folder_name}\\watermark\\{os.path.splitext(os.path.basename(watermarked_shpfile_path))[0]}.png'
    Image.fromarray(watermark.astype(bool)).save(output_watermark_path)
    print("水印创建完成，已保存为", output_watermark_path)


    return output_shapefile_path, error


if __name__ == '__main__':
    watermarked_shpfile_path = select_file('select the watermarked shpfile', [('shpfile', '*.shp')])
    print("当前处理的矢量数据为：", os.path.basename(watermarked_shpfile_path))
    original_watermark_path = select_file('select the watermark', [('watermark file', '*.png *.jpg')])

    if 'building' in os.path.basename(watermarked_shpfile_path):
        vr = np.array([118.31111972859136, 31.318772715218326])
    elif 'landuse' in os.path.basename(watermarked_shpfile_path):
        vr = np.array([118.05745389215463, 31.124422724151433])
    elif 'poi' in os.path.basename(watermarked_shpfile_path):
        vr = np.array([13171310.792046145, 3665864.1180849196])
    elif 'road' in os.path.basename(watermarked_shpfile_path):
        vr = np.array([118.24463147035793, 31.162487809141247])
    elif 'waters' in os.path.basename(watermarked_shpfile_path):
        vr = np.array([335298.4706072447, 3461094.277181611])
    elif 'waterway' in os.path.basename(watermarked_shpfile_path):
        vr = np.array([13152325.677522954, 3650971.9577622036])
    else:
        print('vr不存在')
    extract(watermarked_shpfile_path, original_watermark_path, vr)
