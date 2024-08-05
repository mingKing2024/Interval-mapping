# -*- coding: utf-8 -*-
# @Time    : 2024/2/14 22:34
# @Author  :Fivem
# @File    : embed.py
# @Software: PyCharm
# @last modified:2024/2/14 22:34
import os
import random
from decimal import Decimal

import geopandas as gpd
import numpy as np
from PIL import Image

from get_coor import get_coor_nested, get_coor_array
from select_file import select_file
from to_geodataframe import to_geodataframe


def watermark_embed(coor, coor_l, w, n, R):
    """
    对坐标嵌入水印
    :param coor: 变换后的坐标
    :param w: 水印
    :param coor_l: 区域左下角顶点的坐标
    :return:返回嵌入水印的坐标
    """
    x, y = coor
    xl, yl = coor_l
    k = (x - xl) / 2 ** n
    embed_x = xl + w * (R / 2 ** n) + k
    embed_y = y
    return np.vstack([embed_x, embed_y])


def coor_process(coor, argument, dis):
    """
    对坐标进行处理
    :param coor: 需要处理的坐标
    :return: 嵌入水印的坐标
    """
    n, R, W = argument.values()

    random.seed(dis)

    index = random.randint(0, len(W) - 1)  # 抵抗删点、平移、缩放 不抵抗旋转
    w = int(''.join(map(str, W[index])), 2)

    coor_l = coor // R * R
    embed_coor = watermark_embed(coor, coor_l, w, n, R)
    return embed_coor


def coor_group_process(coor_group, argument):
    """
    对坐标组进行处理
    :param coor_group:需要处理的坐标组
    :return: 返回嵌入水印的坐标组
    """
    embed_coor_group = np.array([[], []])
    coor_group = np.array([[Decimal(str(x)) for x in row] for row in coor_group])
    for i in range(coor_group.shape[1]):
        coor = coor_group[:, i]
        if i == coor_group.shape[1] - 1:
            embed_coor = coor[:, np.newaxis]
        else:
            # dis = abs(coor[1] - coor_group[:, i + 1][1])
            dis = round(abs(coor[1] - coor_group[:, i + 1][1]), 9)
            # print(coor[1] ,coor_group[:, i + 1][1],dis)
            embed_coor = coor_process(coor, argument, dis)
            # print(embed_coor)
        embed_coor_group = np.concatenate((embed_coor_group, embed_coor), axis=1)
    return embed_coor_group


def traversal_nested_coor_group(coor_nested, feature_type, argument):
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
        processed_coor_group = coor_group_process(coor_group, argument)
        # 如果要素为多面，则需要满足首位顶点的坐标相同
        if (feature_type == 'MultiPolygon'
                and np.size(processed_coor_group) not in [0, 2]
                and not np.array_equal(processed_coor_group[:, 0], processed_coor_group[:, -1])):
            processed_coor_group[:, -1] = processed_coor_group[:, 0]
        processed_x_nested.append(processed_coor_group[0, :])
        processed_y_nested.append(processed_coor_group[1, :])
    return np.array([processed_x_nested, processed_y_nested], dtype=object)


def traversal_coor_group(coor_nested, shp_type, processed_shpfile, argument):
    """
    对所有要素进行遍历
    :param coor_nested: 所有要素组成的嵌套坐标数组
    :param shp_type: 每个要素类型组成的数组
    :param processed_shpfile: 处理后的shp文件
    :return: processed_shpfile
    """
    # 遍历每个几何要素
    for feature_index in range(coor_nested.shape[1]):
        coor_group = np.vstack(coor_nested[:, feature_index])
        feature_type = shp_type[feature_index]
        # 判断是否是多线、多面等的情况
        if isinstance(coor_group[0, 0], np.ndarray):
            processed_coor_group = traversal_nested_coor_group(coor_group, feature_type, argument)
        else:
            processed_coor_group = coor_group_process(coor_group, argument)
            # 如果要素为面，则需要满足首尾顶点的坐标相同
            if (feature_type == 'Polygon'
                    and np.size(processed_coor_group) not in [0, 2]
                    and not np.array_equal(processed_coor_group[:, 0], processed_coor_group[:, -1])):
                processed_coor_group[:, -1] = processed_coor_group[:, 0]
        # 将改变的要素坐标组更新到geodataframe
        processed_shpfile = to_geodataframe(processed_shpfile, feature_index, processed_coor_group,
                                            shp_type[feature_index])
    return processed_shpfile


def embed(shpfile_path, watermark_path):
    # -------------------------预定义--------------------------------
    n = 4  # 嵌入强度
    tau = 10 ** (-6)  # 精度容差
    R = Decimal('1e-7')

    # -------------------------数据读取--------------------------------
    original_shpfile = gpd.read_file(shpfile_path)
    coor_nested, feature_type = get_coor_nested(original_shpfile)
    watermark = np.array(Image.open(watermark_path)).astype(int)

    # -------------------------数据预处理--------------------------------
    # 对水印图像需要压缩,将位置探测图案删除
    replace_matrix = np.full((8, 8), -1)
    watermark[:8, :8] = watermark[:8, -8:] = watermark[-8:, :8] = replace_matrix
    watermark = watermark.flatten()  # 将水印图像矩阵转为一维数组
    watermark = list(filter(lambda x: x != -1, watermark))  # 将值为-1的值删除
    watermark += [0] * ((n - len(watermark) % n) % n)  # 对长度不足n的子数组补0
    # 对一维数组进行分组，每n个为一组
    W = np.array_split(watermark, len(watermark) // n)
    argument = {'n': n, 'R': R, 'W': W}

    # -------------------------水印嵌入--------------------------------
    # Go through each object
    watermarked_shpfile = original_shpfile.copy()
    watermarked_shpfile = traversal_coor_group(coor_nested, feature_type, watermarked_shpfile, argument)

    # -------------------------数据输出--------------------------------
    # Create folder
    if not os.path.exists('..\\embed'):
        os.makedirs('..\\embed')
    output_shapefile_path = f'..\\embed\\{os.path.splitext(os.path.basename(watermark_path))[0] + os.path.basename(shpfile_path)}'
    # Save GeoDataFrame as Shapefile
    watermarked_shpfile.to_file(output_shapefile_path)

    watermarked_shpfile = gpd.read_file(output_shapefile_path)
    coor_nested, feature_type = get_coor_nested(watermarked_shpfile)
    coor_array = get_coor_array(coor_nested, feature_type)  # 将嵌套坐标数组合并成一个数组
    vr = np.mean(coor_array, axis=1)
    vr = [float(format(coor, '.15f')) for coor in vr]
    print(vr)

    print("Shapefile创建完成，已保存为", output_shapefile_path)
    return output_shapefile_path, vr


if __name__ == "__main__":
    # -------------------------数据对比--------------------------------
    shpfile_path = select_file("select shpfile", [("shpfile", "*.shp")])
    print("当前处理的矢量数据为：", os.path.basename(shpfile_path))
    watermark_path = select_file('select the watermark', [('watermark file', '*.png *.jpg')])
    embed(shpfile_path, watermark_path)
