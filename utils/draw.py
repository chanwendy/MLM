import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
import os

#直线方程函数
def f_1(x, A, B):
    return A*x + B

def scatter_plot(data,epoch=0,name=None, title=None, x_label="x_label", y_label="y_label", color_point="red", color_line="blue"):
    plt.figure()

    x0 = []
    y0 = []
    for _, i in data.items():
        x0.append(i[0])
        y0.append(i[1])

    #绘制散点
    plt.scatter(x0[:], y0[:], 25, color_point)

    #直线拟合与绘制
    A1, B1 = optimize.curve_fit(f_1, x0, y0)[0]
    x1 = np.arange(0, 1, 0.005)
    y1 = A1*x1 + B1
    plt.plot(x1, y1, color_line)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)

    plt.show()
    root_path = os.path.join("/data/MING/data/NIH/result/figure", "analysis_figure", name)
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    save_path = os.path.join(root_path, str(epoch))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.savefig(os.path.join(save_path, f'{title}.jpg'))  # TODO


def scatter_plot_multi(data1, data2,epoch, name, title=None, x_label="x_label", y_label="y_label", color1="red", color2="blue"):
    plt.figure()

    # Color1
    x0 = []
    y0 = []
    for _, i in data1.items():
        x0.append(i[0])
        y0.append(i[1])

    #绘制散点
    plt.scatter(x0[:], y0[:], 25, color1)

    #直线拟合与绘制
    A1, B1 = optimize.curve_fit(f_1, x0, y0)[0]
    x1 = np.arange(0, 1, 0.005)
    y1 = A1*x1 + B1
    plt.plot(x1, y1, color1)


    # Color2
    x0 = []
    y0 = []
    for _, i in data2.items():
        x0.append(i[0])
        y0.append(i[1])

    #绘制散点
    plt.scatter(x0[:], y0[:], 25, color2)

    #直线拟合与绘制
    A1, B1 = optimize.curve_fit(f_1, x0, y0)[0]
    x1 = np.arange(0, 1, 0.005)
    y1 = A1*x1 + B1
    # plt.plot(x1, y1, color2)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.show()
    root_path = os.path.join("/data/MING/data/NIH/result/figure", "analysis_figure", name)
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    save_path = os.path.join(root_path, str(epoch))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.savefig(os.path.join(save_path, f'{title}.jpg'))  # todo

