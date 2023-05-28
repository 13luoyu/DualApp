import matplotlib.pyplot as plt
import matplotlib

def draw1(y_sample, y_gradient, t_sample, t_gradient):  # mnist_cnn
    plt.rcParams['figure.figsize'] = (6.2,5.5)
    # x_tick = ["$FNN_{1*100}$", "$FNN_{1*150}$", "$FNN_{1*200}$", "$FNN_{1*250}$"]
    x_tick = ["$CNN_{2-1}$", "$CNN_{2-2}$", "$CNN_{2-3}$", "$CNN_{2-4}$"]
    x = range(len(x_tick))
    # y_sample = [5.104, 5.476, 5.310, 6.362]
    # y_gradient = [5.112, 5.476, 5.311, 6.364]
    # t_sample = [1.65, 2.55, 3.54, 4.45]
    # t_gradient = [4.26, 10.58, 19.27, 31.75]
    
    t_max = max(t_gradient) + 5

    width = 0.3

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.bar([xi - width / 2.0 for xi in x], y_sample, width=width, color='LemonChiffon', alpha=0.8, label="Samp. B", hatch="//")
    plt.bar([xi + width / 2.0 for xi in x], y_gradient, width=width, color='Coral', alpha=0.5, label="Grad. B")
    # plt.ylabel("certified lower bound (1e-2)", fontsize=20)
    for a,b in zip(x, y_sample):
        plt.text(a-width/2.0, b, b, ha="center", va="baseline", fontsize=20, rotation=60)
    for a,b in zip(x, y_gradient):
        plt.text(a+width/2.0, b, b, ha="center", va="baseline", fontsize=20, rotation=60)
    plt.xticks(x, x_tick, fontsize=18)
    plt.ylim(3.0, 7.0)
    plt.yticks(fontsize=20)
    # plt.legend(["Samp. B", "Grad. B"], loc='upper left', fontsize=20)
    
    ax2 = plt.twinx()
    # ax2.set_ylabel("time (s)", fontsize=20)
    ax2.set_ylim([1, t_max])
    plt.yticks(fontsize=20)
    plt.plot(x, t_sample, marker="o", color = "green", linewidth=2.0, label="Samp. T", linestyle='--',markersize=12)
    plt.plot(x, t_gradient, marker="s", color = "blue", linewidth=2.0, label="Grad. T", linestyle='--',markersize=12)
    plt.tight_layout()
    # lines, labels = ax.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # plt.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=20)
    plt.savefig("GD_sample_mnist_cnn.png")
    plt.clf()




def draw2(y_sample, y_gradient, t_sample, t_gradient):  # fashion_mnist_cnn
    plt.rcParams['figure.figsize'] = (6.2,5.5)
    x_tick = ["$CNN_{2-1}$", "$CNN_{2-2}$", "$CNN_{2-3}$", "$CNN_{2-4}$"]
    x = range(len(x_tick))
    # y_sample = [2.525, 2.941, 2.85, 2.812]
    # y_gradient = [2.543, 2.959, 2.866, 2.828]
    # t_sample = [1.0, 2.0, 3.5, 5.0]
    # t_gradient = [0.8, 1.5, 2.0, 2.8]

    # y_sample = [8.934, 8.325, 8.567, 8.24]
    # y_gradient = [8.944, 8.331, 8.571, 8.243]
    # t_sample = [1.74, 2.67, 3.19, 4.39]
    # t_gradient = [4.13, 10.55, 18.95, 31.91]

    t_max = max(t_gradient) + 5
    width = 0.3

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.bar([xi - width / 2.0 for xi in x], y_sample, width=width, color='LemonChiffon', alpha=0.8, label="Samp. B", hatch="//")
    plt.bar([xi + width / 2.0 for xi in x], y_gradient, width=width, color='Coral', alpha=0.5, label="Grad. B")
    # plt.ylabel("certified lower bound (1e-2)", fontsize=20)
    for a,b in zip(x, y_sample):
        plt.text(a-width/2.0, b, b, ha="center", va="baseline", fontsize=20, rotation=60)
    for a,b in zip(x, y_gradient):
        plt.text(a+width/2.0, b, b, ha="center", va="baseline", fontsize=20, rotation=60)
    plt.xticks(x, x_tick, fontsize=18)
    plt.ylim(7.0, 9.5)
    plt.yticks(fontsize=20)
    # plt.legend(["Samp. B", "Grad. B"], loc='upper left', fontsize=20)
    
    ax2 = plt.twinx()
    ax2.set_ylabel("Time (s)", fontsize=20)
    ax2.set_ylim([1, t_max])
    plt.yticks(fontsize=20)
    plt.plot(x, t_sample, marker="o", color = "green", linewidth=2.0, label="Samp. T", linestyle='--',markersize=12)
    plt.plot(x, t_gradient, marker="s", color = "blue", linewidth=2.0, label="Grad. T", linestyle='--',markersize=12)
    plt.tight_layout()
    # lines, labels = ax.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # plt.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=20)
    plt.savefig("GD_sample_fashion_mnist_cnn.png")
    plt.clf()



def draw3(y_sample, y_gradient, t_sample, t_gradient):  # mnist_fnn
    plt.rcParams['figure.figsize'] = (6.2,5.5)
    x_tick = ["$FNN_{1*100}$", "$FNN_{1*150}$", "$FNN_{1*200}$", "$FNN_{1*250}$"]
    x = range(len(x_tick))
    # y_sample = [2.526, 2.942, 2.851, 2.814]
    # y_gradient = [2.543, 2.959, 2.866, 2.828]
    # t_sample = [4.35, 6.24, 7.85, 9.62]
    # t_gradient = [1.00, 2.02, 3.38, 5.22]

    width = 0.3

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.bar([xi - width / 2.0 for xi in x], y_sample, width=width, color='LemonChiffon', alpha=0.8, label="Samp. B", hatch="//")
    plt.bar([xi + width / 2.0 for xi in x], y_gradient, width=width, color='Coral', alpha=0.5, label="Grad. B")
    plt.ylabel("Certified lower bound (1e-2)", fontsize=20)
    for a,b in zip(x, y_sample):
        plt.text(a-width/2.0, b, b, ha="center", va="baseline", fontsize=20, rotation=60)
    for a,b in zip(x, y_gradient):
        plt.text(a+width/2.0, b, b, ha="center", va="baseline", fontsize=20, rotation=60)
    plt.xticks(x, x_tick, fontsize=18)
    plt.ylim(1.0, 5.5)
    plt.yticks(fontsize=20)
    # plt.legend(["Samp. B", "Grad. B"], loc='upper left', fontsize=20)
    
    ax2 = plt.twinx()
    # ax2.set_ylabel("time (s)", fontsize=20)
    ax2.set_ylim([0, 15])
    plt.yticks(fontsize=20)
    plt.plot(x, t_sample, marker="o", color = "green", linewidth=2.0, label="Samp. T", linestyle='--',markersize=12)
    plt.plot(x, t_gradient, marker="s", color = "blue", linewidth=2.0, label="Grad. T", linestyle='--',markersize=12)
    plt.tight_layout()
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=20)
    plt.savefig("GD_sample_mnist_fnn.png")
    plt.clf()


def draw4(y_sample, y_gradient, t_sample, t_gradient):  # fashion_mnist_fnn
    plt.rcParams['figure.figsize'] = (6.2,5.5)
    x_tick = ["$FNN_{1*100}$", "$FNN_{1*150}$", "$FNN_{1*200}$", "$FNN_{1*250}$"]
    x = range(len(x_tick))
    # y_sample = [3.379, 3.801, 3.832, 3.894]
    # y_gradient = [3.398, 3.817, 3.847, 3.91]
    # t_sample = [4.03, 6.09, 8.13, 10.00]
    # t_gradient = [1.04, 2.14, 3.63, 5.41]

    width = 0.3

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.bar([xi - width / 2.0 for xi in x], y_sample, width=width, color='LemonChiffon', alpha=0.8, label="Samp. B", hatch="//")
    plt.bar([xi + width / 2.0 for xi in x], y_gradient, width=width, color='Coral', alpha=0.5, label="Grad. B")
    # plt.ylabel("certified lower bound (1e-2)", fontsize=20)
    for a,b in zip(x, y_sample):
        plt.text(a-width/2.0, b, b, ha="center", va="baseline", fontsize=20, rotation=60)
    for a,b in zip(x, y_gradient):
        plt.text(a+width/2.0, b, b, ha="center", va="baseline", fontsize=20, rotation=60)
    plt.xticks(x, x_tick, fontsize=18)
    plt.ylim(2.0, 4.5)
    plt.yticks(fontsize=20)
    # plt.legend(["Samp. B", "Grad. B"], loc='upper left', fontsize=20)
    
    ax2 = plt.twinx()
    # ax2.set_ylabel("time (s)", fontsize=20)
    ax2.set_ylim([0, 14])
    plt.yticks(fontsize=20)
    plt.plot(x, t_sample, marker="o", color = "green", linewidth=2.0, label="Samp. T", linestyle='--',markersize=12)
    plt.plot(x, t_gradient, marker="s", color = "blue", linewidth=2.0, label="Grad. T", linestyle='--',markersize=12)
    plt.tight_layout()
    # lines, labels = ax.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # plt.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=20)
    plt.savefig("GD_sample_fashion_mnist_fnn.png")
    plt.clf()


if __name__ == "__main__":
    y_bound, y_time = [], []
    with open("../logs/run_figure_10.log", "r") as f:
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
    y_bound = [round(y, 3) for y in y_bound]
    draw1(y_bound[0:4], y_bound[4:8], y_time[0:4], y_time[4:8])
    draw2(y_bound[8:12], y_bound[12:16], y_time[8:12], y_time[12:16])
    draw3(y_bound[16:20], y_bound[20:24], y_time[16:20], y_time[20:24])
    draw4(y_bound[24:28], y_bound[28:32], y_time[24:28], y_time[28:32])