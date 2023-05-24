import matplotlib.pyplot as plt
import numpy as np


def draw():
    x = [i for i in range(100, 2000, 100)]
    y_bound = [1.287, 1.288, 1.289, 1.29, 1.29, 1.29, 1.291, 1.291, 1.291, 1.291, 1.292, 1.292, 1.292, 1.292, 1.292, 1.292, 1.292, 1.292, 1.292]
    y_time = [0.728, 1.576, 2.424, 3.272, 4.12, 4.968, 5.816, 6.664, 7.512, 8.36, 9.207, 10.055, 10.903, 11.751, 12.599, 13.447, 14.295, 15.143, 15.991]

    plt.rcParams['figure.figsize'] = (7,6)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(x, y_bound, marker="s", color="blue", linewidth=2.0, label="Certified bound", linestyle="-", markersize=12)
    plt.ylim(1.286, 1.293)
    plt.yticks(fontsize=20)
    plt.xticks([500, 1000, 1500], fontsize=20)
    plt.xlabel("Sample number", fontsize=20)
    plt.ylabel("Certified lower bound (1e-2)", fontsize=20)

    ax2=plt.twinx()
    ax2.set_ylim(0, 17)
    plt.yticks(fontsize=20)
    plt.plot(x, y_time, marker="*", color="green", linewidth=2.0, label="Time", linestyle="-", markersize=12)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc='lower right', fontsize=20)

    plt.tight_layout()
    plt.savefig("Sampling_fashion_mnist_fnn.png")

def draw2():
    x = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
    y_bound = [3.879, 3.886, 3.892, 3.898, 3.901, 3.904,3.906,3.908,3.909,3.909, 3.908, 3.906,3.902,3.899,3.894,3.886,3.88,3.871,3.86,3.849]
    y_time = [4.498,4.401,4.426,4.755,4.791,4.786,4.771,4.53,4.511,4.583,4.566,4.574,4.569,4.572,4.561,4.564,4.4,4.566,4.524,4.502]

    plt.rcParams['figure.figsize'] = (7,6)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(x, y_bound, marker="s", color="blue", linewidth=2.0, label="Certified bound", linestyle="-", markersize=12)
    plt.ylim(3.847, 3.913)
    plt.yticks(fontsize=20)
    plt.xticks([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
    plt.xlabel("Step length", fontsize=20)
    # plt.ylabel("certified lower bound (1e-2)", fontsize=20)

    ax2=plt.twinx()
    ax2.set_ylim(0, 6)
    plt.yticks(fontsize=20)
    plt.plot(x, y_time, marker="*", color="green", linewidth=2.0, label="Time", linestyle="-", markersize=12)
    plt.ylabel("Time (s)", fontsize=20)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc='lower left', fontsize=20)

    plt.tight_layout()
    plt.savefig("GD_fashion_mnist_fnn.png")



draw()
draw2()