import matplotlib.pyplot as plt
import numpy as np


def draw(y_bound, y_time):
    x = [i for i in range(100, 2000, 100)]
    # y_bound = [3.797, 3.798, 3.799, 3.799, 3.8, 3.8, 3.8, 3.801, 3.801, 3.801, 3.801, 3.801, 3.801, 3.802, 3.802, 3.802, 3.802, 3.802, 3.802]
    # y_time = [0.728, 1.576, 2.424, 3.272, 4.12, 4.968, 5.816, 6.664, 7.512, 8.36, 9.207, 10.055, 10.903, 11.751, 12.599, 13.447, 14.295, 15.143, 15.991]

    plt.rcParams['figure.figsize'] = (7,6)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(x, y_bound, marker="s", color="blue", linewidth=2.0, label="Certified bound", linestyle="-", markersize=12)
    plt.ylim(3.796, 3.803)
    plt.yticks(fontsize=20)
    plt.xticks([500, 1000, 1500], fontsize=20)
    plt.xlabel("Sample number", fontsize=20)
    plt.ylabel("Certified lower bound (1e-2)", fontsize=20)

    ax2=plt.twinx()
    ax2.set_ylim(0, 12)
    plt.yticks([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0], fontsize=20)
    plt.plot(x, y_time, marker="*", color="green", linewidth=2.0, label="Time", linestyle="-", markersize=12)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc='lower right', fontsize=20)

    plt.tight_layout()
    plt.savefig("Sampling_fashion_mnist_fnn.png")

def draw2(y_bound, y_time):
    x = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
    # y_bound = [3.787, 3.793, 3.799, 3.804, 3.808, 3.811,3.814,3.816,3.817,3.817, 3.816, 3.814,3.812,3.808,3.804,3.798,3.791,3.784,3.775,3.765]
    # y_time = [4.498,4.401,4.426,4.755,4.791,4.786,4.771,4.53,4.511,4.583,4.566,4.574,4.569,4.572,4.561,4.564,4.4,4.566,4.524,4.502]

    plt.rcParams['figure.figsize'] = (7,6)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(x, y_bound, marker="s", color="blue", linewidth=2.0, label="Certified bound", linestyle="-", markersize=12)
    plt.ylim(3.76, 3.82)
    plt.yticks([3.765, 3.775, 3.785, 3.795, 3.805, 3.815], fontsize=20)
    plt.xticks([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
    plt.xlabel("Step length", fontsize=20)
    # plt.ylabel("certified lower bound (1e-2)", fontsize=20)

    ax2=plt.twinx()
    ax2.set_ylim(0, 4)
    plt.yticks([0.0, 1.0, 2.0, 3.0, 4.0], fontsize=20)
    plt.plot(x, y_time, marker="*", color="green", linewidth=2.0, label="Time", linestyle="-", markersize=12)
    plt.ylabel("Time (s)", fontsize=20)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc='lower left', fontsize=20)

    plt.tight_layout()
    plt.savefig("GD_fashion_mnist_fnn.png")


if __name__ == "__main__":
    y_bound, y_time = [], []
    with open("../logs/run_figure_9.log", "r") as f:
        lines = f.readlines()
    l0 = 0
    for line in lines:
        if "[L0]" in line:
            ss = line.split(" ")
            if l0 == 0:
                runtime = 0.0
                for i, s in enumerate(ss):
                    if s == "robustness":
                        y_bound.append(100 * float(ss[i+2][:-1]))
                    elif s == "runtime":
                        runtime += float(ss[i+2][:-1])
                y_time.append(runtime)
                l0 = 1
            elif l0 == 1:
                l0 = 0
    # print(y_bound, y_time)
    draw(y_bound[0:19], y_time[0:19])
    draw2(y_bound[20:40], y_time[20:40])