# _*_ coding: utf-8 _*_
import os
import matplotlib.pyplot as plt


def plot_figures(datas: list, title: str, labels: list, save_path):
    x = range(len(datas))
    y = datas
    plt.title(title)
    plt.xlabel(labels[0])  # x轴标签
    plt.ylabel(labels[1])  # y轴标签
    plt.plot(x, y)
    plt.savefig(fname=save_path, dpi=300)
