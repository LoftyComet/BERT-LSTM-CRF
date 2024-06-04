"""
一些工具函数
"""
import pickle
import pyvista as pv

def merge_maps(dict1, dict2):
    """用于合并两个word2id或者两个tag2id"""
    for key in dict2.keys():
        if key not in dict1:
            dict1[key] = len(dict1)
    return dict1


def save_model(model, file_name):
    """用于保存模型"""
    with open(file_name, "wb") as f:
        pickle.dump(model, f)


def load_model(file_name):
    """用于加载模型"""
    with open(file_name, "rb") as f:
        model = pickle.load(f)
    return model


# LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
# 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
def extend_maps(word2id, tag2id, for_crf=True):
    word2id['<unk>'] = len(word2id)
    word2id['<pad>'] = len(word2id)
    tag2id['<unk>'] = len(tag2id)
    tag2id['<pad>'] = len(tag2id)
    # 如果是加了CRF的bilstm  那么还要加入<start> 和 <end>token
    if for_crf:
        word2id['<start>'] = len(word2id)
        word2id['<end>'] = len(word2id)
        tag2id['<start>'] = len(tag2id)
        tag2id['<end>'] = len(tag2id)

    return word2id, tag2id


def prepocess_data_for_lstmcrf(word_lists, tag_lists, test=False):
    """预处理lstm-crf数据"""
    assert len(word_lists) == len(tag_lists)
    for i in range(len(word_lists)):
        word_lists[i].append("<end>")
        if not test:  # 如果是测试数据，就不需要加end token了
            tag_lists[i].append("<end>")

    return word_lists, tag_lists


def flatten_lists(lists):
    flatten_list = []
    for l in lists:
        if type(l) == list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list


import matplotlib.pyplot as plt
import numpy as np


def map_to_color(value, vmin, vmax, cmap_name='jet'):
    """
    将数字映射到颜色。

    参数：
    - value: 要映射的数字值
    - vmin: 数据中的最小值
    - vmax: 数据中的最大值
    - cmap_name: 要使用的颜色映射名称，默认为'viridis'

    返回值：
    - color: 映射到的颜色值，格式为(R, G, B, A)，每个通道的值在0到1之间
    """
    cmap = plt.get_cmap(cmap_name)  # 获取颜色映射对象
    norm = plt.Normalize(vmin, vmax)  # 创建归一化器

    # 将值映射到[0, 1]范围
    normalized_value = norm(value)

    # 根据归一化后的值获取对应的颜色
    color = cmap(normalized_value)

    return color


def draw_point(pred_tag_lists, test_tag_lists):
    p = pv.Plotter()
    # 预测点
    for qq in range(len(pred_tag_lists)):
        point = [pred_tag_lists[qq][0], pred_tag_lists[qq][1], pred_tag_lists[qq][2]]
        mesh = pv.PolyData(point)  # PolyData对象的实例化
        p.add_mesh(mesh, color=map_to_color(qq, 0, len(test_tag_lists)), point_size=5)

    # 目标球
    for q1 in range(len(test_tag_lists)):
        sphere = pv.Sphere(radius=0.02, center=(test_tag_lists[q1][0], test_tag_lists[q1][1], test_tag_lists[q1][2]))
        p.add_mesh(sphere, color=map_to_color(q1, 0, len(test_tag_lists)), opacity=0.5)

    p.camera_position = 'xy'
    p.show_grid()
    p.show(cpos="xy")


def draw_points(train_tag_lists):
    x_count = [0] * len(train_tag_lists)
    x_list = []
    points = []
    p = pv.Plotter()
    # 预测点
    for qq in range(len(train_tag_lists)):

        flag = -1
        for i, x in enumerate(x_list):
            if abs(float(list(train_tag_lists[qq])[0]) - float(x[0])) < 0.00001:
                flag = i
                break
        if flag != -1:
            x_count[flag] += 1

        else:
            x_count[len(x_list)] = 1
            x_list.append(list(train_tag_lists[qq]))

    x_count = x_count[0:len(x_list)]
    maxvalue = max(x_count)
    for i, x in enumerate(x_count):
        if x > (maxvalue / 6):
            x_count[i] = maxvalue / 6

    for i, key in enumerate(x_list):
        point = [key[0], key[1], -key[2]]
        points.append(point)
    mesh = pv.PolyData(points)  # PolyData对象的实例化
    p.add_mesh(mesh, scalars=x_count, cmap='bwr', point_size=5)


    p.camera_position = 'xy'
    p.show_grid()
    p.show(cpos="xy")
    print(x_count)

def draw_finger(x, y, z, loc):
    p = pv.Plotter()
    # 预测点
    for qq in range(len(x)):
        point = [x[qq], y[qq], z[qq]]
        # mesh = pv.PolyData(point)  # PolyData对象的实例化
        # p.add_mesh(mesh, color=map_to_color(qq, 0, len(x)), point_size=5)

        sphere = pv.Sphere(radius=0.005, center=point)
        p.add_mesh(sphere, opacity=0.5)

    sphere = pv.Sphere(radius=0.02, center=loc)
    p.add_mesh(sphere, opacity=0.5)
    p.camera_position = 'yx'
    p.show_grid()
    p.show(cpos="yz")


def draw_error(pred_tag_lists, test_tag_lists, distances):
    p = pv.Plotter()

    # 自定义坐标轴
    p.add_axes(
        interactive=True,
        line_width=2,
        color='black',
        x_color='red',  # 设置X轴颜色
        y_color='green',  # 设置Y轴颜色
        z_color='blue',  # 设置Z轴颜色
        xlabel='X Axis',  # 设置X轴标签
        ylabel='Y Axis',  # 设置Y轴标签
        zlabel='Z Axis'  # 设置Z轴标签
    )

    points = []
    cmap = 'bwr'
    for index, value in enumerate(distances):
        if value > 0.10:
            distances[index] = 0.10

    # 目标点
    for qq in range(len(test_tag_lists)):
        point = [test_tag_lists[qq][0], test_tag_lists[qq][1], -test_tag_lists[qq][2]]
        points.append(point)
    mesh = pv.PolyData(points)  # PolyData对象的实例化
    p.add_mesh(mesh, scalars=distances, cmap=cmap, point_size=5)
    p.show_grid()
    p.camera_position = 'xy'
    p.show(cpos="xy")
