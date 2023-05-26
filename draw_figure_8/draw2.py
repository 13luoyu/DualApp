import matplotlib.pyplot as plt
import numpy as np

def draw1(file_name, xtick=None):
    plt.rcParams['figure.figsize'] = (6.3,4.7)
    datas = []
    with open(file_name, "r") as f:
        lines = [l.strip() for l in f.readlines()]
    state = 0
    data = []
    for l in lines:
        if l == '':
            state += 1
            datas.append(data)
            data = []
        elif state == 0:
            eps = float(l)
            data.append(eps)
        else:
            impr = int(l)
            data.append(impr)
    datas.append(data)

    Listcolors = ['darkred', 'darkred', 'green', 'green', 'blue', 'blue', 'darkred', 'black']
    Markers = ['o', 'o', 's', 's', '*', '*', '*', '+', 'x']  # 对第一个图
    # Listcolors = ['darkred', 'green', 'blue', 'darkred', 'green', 'blue', 'darkred', 'black']
    # Markers = ['o', 's', '*', 'o', 's', '*', '*', '+', 'x']  # 对其他模型
    # Listcolors = ['darkred', 'green', 'blue', 'darkred', 'blue', 'darkred', 'black']
    # Markers = ['o', 's', '*', 'o', '*', '*', '+', 'x']  # 对tanh conv模型
    for i, data in enumerate(datas):
        if i == 0:
            continue
        if i % 2 == 1:
            plt.plot(datas[0], datas[i], marker=Markers[i-1], color=Listcolors[i-1], linewidth=2.0, linestyle='-',markersize=12)
        else:
            plt.plot(datas[0], datas[i], marker=Markers[i-1], color=Listcolors[i-1], linewidth=2.0, linestyle='--',markersize=12)
    # plt.xlabel("epsilon", fontsize=30)
    plt.ylabel("Verif. ratio (%)", fontsize=30)  # 对第一列图
    if xtick is not None:
        plt.xticks(xtick, fontsize=30)
    else:
        plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(["DA,s","DA,t","αβ,s","αβ,t","ER,s","ER,t"], loc='lower left', fontsize=23)  # 对第一个图
    plt.tight_layout()
    plt.savefig(f"{file_name[:-4]}.png")
    plt.clf()


def draw2(file_name, xtick=None):
    plt.rcParams['figure.figsize'] = (6,4.7)
    datas = []
    with open(file_name, "r") as f:
        lines = [l.strip() for l in f.readlines()]
    state = 0
    data = []
    for l in lines:
        if l == '':
            state += 1
            datas.append(data)
            data = []
        elif state == 0:
            eps = float(l)
            data.append(eps)
        else:
            impr = int(l)
            data.append(impr)
    datas.append(data)

    # Listcolors = ['darkred', 'darkred', 'green', 'green', 'blue', 'blue', 'darkred', 'black']
    # Markers = ['o', 'o', 's', 's', '*', '*', '*', '+', 'x']  # 对第一个图
    Listcolors = ['darkred', 'green', 'blue', 'darkred', 'green', 'blue', 'darkred', 'black']
    Markers = ['o', 's', '*', 'o', 's', '*', '*', '+', 'x']  # 对其他模型
    # Listcolors = ['darkred', 'green', 'blue', 'darkred', 'blue', 'darkred', 'black']
    # Markers = ['o', 's', '*', 'o', '*', '*', '+', 'x']  # 对tanh conv模型
    for i, data in enumerate(datas):
        if i == 0:
            continue
        if i <= 3:
            plt.plot(datas[0], datas[i], marker=Markers[i-1], color=Listcolors[i-1], linewidth=2.0, linestyle='-',markersize=12)
        else:
            plt.plot(datas[0], datas[i], marker=Markers[i-1], color=Listcolors[i-1], linewidth=2.0, linestyle='--',markersize=12)
    # plt.xlabel("epsilon", fontsize=30)
    # plt.ylabel("verif. ratio(%)", fontsize=30)  # 对第一列图
    if xtick is not None:
        plt.xticks(xtick, fontsize=30)
    else:
        plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    # plt.legend(["DA,s","DA,t","αβ,s","αβ,t","ER,s","ER,t"], loc='lower left', fontsize=23)  # 对第一个图
    plt.tight_layout()
    plt.savefig(f"{file_name[:-4]}.png")
    plt.clf()


def draw3(file_name, xtick=None):
    plt.rcParams['figure.figsize'] = (6,4.7)
    datas = []
    with open(file_name, "r") as f:
        lines = [l.strip() for l in f.readlines()]
    state = 0
    data = []
    for l in lines:
        if l == '':
            state += 1
            datas.append(data)
            data = []
        elif state == 0:
            eps = float(l)
            data.append(eps)
        else:
            impr = int(l)
            data.append(impr)
    datas.append(data)

    # Listcolors = ['darkred', 'darkred', 'green', 'green', 'blue', 'blue', 'darkred', 'black']
    # Markers = ['o', 'o', 's', 's', '*', '*', '*', '+', 'x']  # 对第一个图
    # Listcolors = ['darkred', 'green', 'blue', 'darkred', 'green', 'blue', 'darkred', 'black']
    # Markers = ['o', 's', '*', 'o', 's', '*', '*', '+', 'x']  # 对其他模型
    Listcolors = ['darkred', 'green', 'blue', 'darkred', 'blue', 'darkred', 'black']
    Markers = ['o', 's', '*', 'o', '*', '*', '+', 'x']  # 对tanh conv模型
    for i, data in enumerate(datas):
        if i == 0:
            continue
        if i <= 3:
            plt.plot(datas[0], datas[i], marker=Markers[i-1], color=Listcolors[i-1], linewidth=2.0, linestyle='-',markersize=12)
        else:
            plt.plot(datas[0], datas[i], marker=Markers[i-1], color=Listcolors[i-1], linewidth=2.0, linestyle='--',markersize=12)
    # plt.xlabel("epsilon", fontsize=30)
    # plt.ylabel("verif. ratio(%)", fontsize=30)  # 对第一列图
    if xtick is not None:
        plt.xticks(xtick, fontsize=30)
    else:
        plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    # plt.legend(["DA,s","DA,t","αβ,s","αβ,t","ER,s","ER,t"], loc='lower left', fontsize=23)  # 对第一个图
    plt.tight_layout()
    plt.savefig(f"{file_name[:-4]}.png")
    plt.clf()


def draw4(file_name, xtick=None):
    plt.rcParams['figure.figsize'] = (6.3,4.7)
    datas = []
    with open(file_name, "r") as f:
        lines = [l.strip() for l in f.readlines()]
    state = 0
    data = []
    for l in lines:
        if l == '':
            state += 1
            datas.append(data)
            data = []
        elif state == 0:
            eps = float(l)
            data.append(eps)
        else:
            impr = int(l)
            data.append(impr)
    datas.append(data)

    # Listcolors = ['darkred', 'darkred', 'green', 'green', 'blue', 'blue', 'darkred', 'black']
    # Markers = ['o', 'o', 's', 's', '*', '*', '*', '+', 'x']  # 对第一个图
    Listcolors = ['darkred', 'green', 'blue', 'darkred', 'green', 'blue', 'darkred', 'black']
    Markers = ['o', 's', '*', 'o', 's', '*', '*', '+', 'x']  # 对其他模型
    # Listcolors = ['darkred', 'green', 'blue', 'darkred', 'blue', 'darkred', 'black']
    # Markers = ['o', 's', '*', 'o', '*', '*', '+', 'x']  # 对tanh conv模型
    for i, data in enumerate(datas):
        if i == 0:
            continue
        if i <= 3:
            plt.plot(datas[0], datas[i], marker=Markers[i-1], color=Listcolors[i-1], linewidth=2.0, linestyle='-',markersize=12)
        else:
            plt.plot(datas[0], datas[i], marker=Markers[i-1], color=Listcolors[i-1], linewidth=2.0, linestyle='--',markersize=12)
    # plt.xlabel("epsilon", fontsize=30)
    plt.ylabel("Verif. ratio (%)", fontsize=30)  # 对第一列图
    if xtick is not None:
        plt.xticks(xtick, fontsize=30)
    else:
        plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    # plt.legend(["DA,s","DA,t","αβ,s","αβ,t","ER,s","ER,t"], loc='lower left', fontsize=23)  # 对第一个图
    plt.tight_layout()
    plt.savefig(f"{file_name[:-4]}.png")
    plt.clf()



if __name__ == '__main__':
    draw2("mnist_fc_pgd1.txt", xtick=[0.02,0.04,0.06])
    draw2("mnist_fc_pgd2.txt", xtick=[0.04,0.06,0.08,0.1])
    draw2("cifar_fc_pgd1.txt", xtick=[0.002,0.004,0.006])
    draw2("cifar_fc_pgd2.txt", xtick=[0.004,0.008,0.012])