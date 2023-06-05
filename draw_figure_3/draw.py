import matplotlib.pyplot as plt
import numpy as np

def draw1():
    # Draw the average overestimation for neural network defined in Figure 2
    plt.rcParams['figure.figsize'] = (6,5)
    x_tick = ["$x_1$", "$x_2$", "$x_3$", "$x_4$", "$x_5$", "$x_6$", "$y_1$", "$y_2$"]
    x = range(len(x_tick))

    y1_real = [-1.0,-1.0,0.119,0.119,0.702,0.434,0.128,-1.301]
    y1_NW = [-1.0,-1.0,-2.0,-2.0,0.687,-0.432,0.045,-1.392]
    y1_DC = [-1.0,-1.0,-2.0,-2.0,0.666,-0.398,0.04,-1.364]
    y1_VN = [-1.0,-1.0,-2.0,-2.0,0.666,-0.398,0.035,-1.353]
    y1_RV = [-1.0,-1.0,-2.0,-2.0,0.023,-0.858,-0.138,-1.51]
    y2_real = [1.0,1.0,0.881,0.881,0.959,0.78,0.392,-0.924]
    y2_NW = [1.0,1.0,2.0,2.0,3.313,1.432,0.494,-0.758]
    y2_DC = [1.0,1.0,2.0,2.0,3.333,1.398,0.429,-0.8]
    y2_VN = [1.0,1.0,2.0,2.0,3.334,1.398,0.425,-0.796]
    y2_RV = [1.0,1.0,2.0,2.0,3.977,1.858,0.564,-0.51]

    diff_NW, diff_DC, diff_VN, diff_RV = [], [], [], []
    for i, xi in enumerate(x):
        diff_NW.append(((y2_NW[i] - y1_NW[i]) - (y2_real[i] - y1_real[i])) / (y2_real[i] - y1_real[i]))
        diff_DC.append(((y2_DC[i] - y1_DC[i]) - (y2_real[i] - y1_real[i])) / (y2_real[i] - y1_real[i]))
        diff_VN.append(((y2_VN[i] - y1_VN[i]) - (y2_real[i] - y1_real[i])) / (y2_real[i] - y1_real[i]))
        diff_RV.append(((y2_RV[i] - y1_RV[i]) - (y2_real[i] - y1_real[i])) / (y2_real[i] - y1_real[i]))


    width = 0.2
    plt.bar([xi - width * 1.5 for xi in x[2:]], [c * 100 for c in diff_NW[2:]], width=width)
    plt.bar([xi - width * 0.5 for xi in x[2:]], [c * 100 for c in diff_DC[2:]], width=width)
    plt.bar([xi + width * 0.5 for xi in x[2:]], [c * 100 for c in diff_VN[2:]], width=width)
    plt.bar([xi + width * 1.5 for xi in x[2:]], [c * 100 for c in diff_RV[2:]], width=width)
    plt.xticks(x[2:], x_tick[2:], fontsize=20)
    plt.yticks(fontsize=20)
    # plt.ylim(0, 1000)
    plt.ylabel("Diff (%)", fontsize=20)
    plt.tight_layout()
    plt.legend(["NW","DC", "VN", "RV"], loc="upper left", fontsize=18)

    plt.savefig("diff_case.png")
    plt.clf()



def draw2():
    # Draw the average overestimation for all 50000 cases
    plt.rcParams['figure.figsize'] = (6,5)
    x_tick = ["$x_1$", "$x_2$", "$x_3$", "$x_4$", "$x_5$", "$x_6$", "$y_1$", "$y_2$"]
    x = range(len(x_tick))

    y1_real = [-0.99897102, -0.99817692,  0.11344028,  0.11398042,  0.34434016,  0.34834904, -0.34414732, -0.3470379 ]
    y1_NW = [-0.99897275, -0.99817904, -2.9967212,  -2.99507348, -1.09026272, -1.06549861, -0.45121916, -0.45466077]
    y1_DC = [-0.99897275, -0.99817904, -2.9967212,  -2.99507348, -1.26985681, -1.24364862, -0.55166396, -0.5555051 ]
    y1_VN = [-0.99897275, -0.99817904, -2.9967212,  -2.99507348, -1.2555697,  -1.2292267, -0.54611454, -0.54999543]
    y1_RV = [-0.99897275, -0.99817904, -2.9967212,  -2.99507348, -1.77704776, -1.75416687, -0.97834783, -0.98114295]
    y2_real = [1.00102898, 1.00182308, 0.88558848, 0.88662224, 0.65212708, 0.6560265, 0.3482961,  0.3430355 ]
    y2_NW = [1.00102725, 1.00182096, 2.98828525, 2.99776994, 1.06824401, 1.09019353, 0.45457222, 0.45059402]
    y2_DC = [1.00102725, 1.00182096, 2.98828525, 2.99776994, 1.24997696, 1.27386663, 0.55459543, 0.5491914 ]
    y2_VN = [1.00102725, 1.00182096, 2.98828525, 2.99776994, 1.23620217, 1.25936058, 0.54930848, 0.54361559]
    y2_RV = [1.00102725, 1.00182096, 2.98828525, 2.99776994, 1.75347343, 1.78117948, 0.98190921, 0.97566867]

    diff_NW, diff_DC, diff_VN, diff_RV = [], [], [], []
    for i, xi in enumerate(x):
        diff_NW.append(((y2_NW[i] - y1_NW[i]) - (y2_real[i] - y1_real[i])) / (y2_real[i] - y1_real[i]))
        diff_DC.append(((y2_DC[i] - y1_DC[i]) - (y2_real[i] - y1_real[i])) / (y2_real[i] - y1_real[i]))
        diff_VN.append(((y2_VN[i] - y1_VN[i]) - (y2_real[i] - y1_real[i])) / (y2_real[i] - y1_real[i]))
        diff_RV.append(((y2_RV[i] - y1_RV[i]) - (y2_real[i] - y1_real[i])) / (y2_real[i] - y1_real[i]))


    width = 0.2
    plt.bar([xi - width * 1.5 for xi in x[2:]], diff_NW[2:], width=width)
    plt.bar([xi - width * 0.5 for xi in x[2:]], diff_DC[2:], width=width)
    plt.bar([xi + width * 0.5 for xi in x[2:]], diff_VN[2:], width=width)
    plt.bar([xi + width * 1.5 for xi in x[2:]], diff_RV[2:], width=width)
    plt.xticks(x[2:], x_tick[2:], fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("diff", fontsize=20)
    plt.tight_layout()
    plt.legend(["NW","DC", "VN", "RV"], loc="upper left", fontsize=18)

    plt.savefig("all_case.png")


def compute1():
    # Compute the average intervals of each methods for y1 and y2
    with open("approximation_domain_sigmoid.txt") as f:
        lines = f.readlines()
    y_args = []
    con_args = []
    step = 0
    y1_NW, y1_DC, y1_VN, y1_RV = np.zeros(8), np.zeros(8), np.zeros(8), np.zeros(8)
    y2_NW, y2_DC, y2_VN, y2_RV = np.zeros(8), np.zeros(8), np.zeros(8), np.zeros(8)
    for line in lines:
        # 读权重
        if step == 0:
            if "Weights:" in line:
                step = 1
        elif step == 1 and "NeWise" not in line:
            y_args += line.strip().split(" ")
        elif step == 1 and "NeWise" in line:
            step = 2
        elif step == 2 and "DeepCert" not in line:
            if line == "\n":
                continue
            l = line[4:-2]
            con_args += l.strip().split(" ")
        elif step == 2 and "DeepCert" in line:
            step = 3
        elif step == 3 and "VeriNet" not in line:
            if line == "\n":
                continue
            l = line[4:-2]
            con_args += l.strip().split(" ")
        elif step == 3 and "VeriNet" in line:
            step = 4
        elif step == 4 and "RobustVerifier" not in line:
            if line == "\n":
                continue
            l = line[4:-2]
            con_args += l.strip().split(" ")
        elif step == 4 and "RobustVerifier" in line:
            step = 5
        elif step == 5 and "====" not in line:
            if line == "\n":
                continue
            l = line[4:-2]
            con_args += l.strip().split(" ")
        
        con_args = [float(co) for co in con_args]

        if len(con_args) == 16:
            if step == 2:
                for i in range(0, 16, 2):
                    y1_NW[int(i/2)] += con_args[i]
                    y2_NW[int(i/2)] += con_args[i+1]
            elif step == 3:
                for i in range(0, 16, 2):
                    y1_DC[int(i/2)] += con_args[i]
                    y2_DC[int(i/2)] += con_args[i+1]
            elif step == 4:
                for i in range(0, 16, 2):
                    y1_VN[int(i/2)] += con_args[i]
                    y2_VN[int(i/2)] += con_args[i+1]
            else:
                for i in range(0, 16, 2):
                    y1_RV[int(i/2)] += con_args[i]
                    y2_RV[int(i/2)] += con_args[i+1]
                step = 0
            y_args, con_args = [], []
    print(y1_NW / 50000, y2_NW / 50000)
    print(y1_DC / 50000, y2_DC / 50000)
    print(y1_VN / 50000, y2_VN / 50000)
    print(y1_RV / 50000, y2_RV / 50000)


def compute2():
    # Compute the actual intervals for y1 and y2
    with open("actual_domain_sigmoid.txt") as f:
        lines = f.readlines()
    y1, y2 = np.zeros(8), np.zeros(8)
    data = []
    for line in lines:
        if line == "\n" and len(data) == 16:
            for i in range(0, 16, 2):
                data = [float(d) for d in data]
                y1[int(i/2)] += data[i]
                y2[int(i/2)] += data[i+1]
            data = []
        else:
            data += line.strip().split(" ")
    print(y1 / 50000, y2 / 50000)




def draw3(app_file, act_file):
    with open(app_file) as f:
        lines = f.readlines()
    
    real, NW, DC, VN, RV = [], [], [], [], []
    step = 0
    for line in lines:
        if step == 0:
            if "NeWise" in line:
                step = 1
        elif step == 1:
            if line == "\n":
                continue
            elif "DeepCert" in line:
                step = 2
                continue
            elif "y1" in line or "y2" in line:
                l = line[4:-2]
                ls = l.split(" ")
                NW.append(float(ls[1]) - float(ls[0]))
        elif step == 2:
            if line == "\n":
                continue
            elif "VeriNet" in line:
                step = 3
                continue
            elif "y1" in line or "y2" in line:
                l = line[4:-2]
                ls = l.split(" ")
                DC.append(float(ls[1]) - float(ls[0]))
        elif step == 3:
            if line == "\n":
                continue
            elif "RobustVerifier" in line:
                step = 4
                continue
            elif "y1" in line or "y2" in line:
                l = line[4:-2]
                ls = l.split(" ")
                VN.append(float(ls[1]) - float(ls[0]))
        elif step == 4:
            if line == "\n":
                continue
            elif "=====" in line:
                step = 0
                continue
            elif "y1" in line or "y2" in line:
                l = line[4:-2]
                ls = l.split(" ")
                RV.append(float(ls[1]) - float(ls[0]))
    
    with open(act_file) as f:
        lines = f.readlines()
    i = 0
    for line in lines:
        if line == "\n":
            i = 0
        else:
            i += 1
        if i == 7 or i == 8:
            ls = l.split(" ")
            real.append(float(ls[1]) - float(ls[0]))
    
    total_NW, total_DC, total_VN, total_RV = [], [], [], []
    x = [float(i) / 10.0 for i in range(31)]
    for ii in range(len(x)-1):
        a, b = x[ii], x[ii+1]
        c_NW, c_DC, c_VN, c_RV = 0, 0, 0, 0
        for i in range(len(real)):
            rate_NW = (NW[i] * real[i]) / real[i]
            rate_DC = (DC[i] * real[i]) / real[i]
            rate_VN = (VN[i] * real[i]) / real[i]
            rate_RV = (RV[i] * real[i]) / real[i]
            if a <= rate_NW and rate_NW < b:
                c_NW += 1
            if a <= rate_DC and rate_DC < b:
                c_DC += 1
            if a <= rate_VN and rate_VN < b:
                c_VN += 1
            if a <= rate_RV and rate_RV < b:
                c_RV += 1
        total_NW.append(c_NW)
        total_DC.append(c_DC)
        total_VN.append(c_VN)
        total_RV.append(c_RV)
    print(total_NW, sum(total_NW))
    print(total_DC, sum(total_DC))
    print(total_VN, sum(total_VN))
    print(total_RV, sum(total_RV))

    plt.rcParams['figure.figsize'] = (6,5)
    x = [xi * 100 for xi in x[:-1]]
    total_NW = [float(n) / 1000.0 for n in total_NW]
    total_DC = [float(n) / 1000.0 for n in total_DC]
    total_VN = [float(n) / 1000.0 for n in total_VN]
    total_RV = [float(n) / 1000.0 for n in total_RV]
    plt.plot(x, total_NW, linewidth=2.0, linestyle="-")
    plt.plot(x, total_DC, linewidth=2.0, linestyle="-")
    plt.plot(x, total_VN, linewidth=2.0, linestyle="-")
    plt.plot(x, total_RV, linewidth=2.0, linestyle="-")
    plt.legend(["NW", "DC", "VN", "RV"], loc="upper right", fontsize=20)
    plt.ylabel("Num. ratio (%)", fontsize=20)
    plt.xlabel("Diff. (%)", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig("percentage.png")



if __name__ == "__main__":
    # compute1()
    # compute2()
    draw1()  # draw Figure 3(a)
    # draw2()
    draw3("../logs/figure_3_approximation_domain.txt", "../logs/figure_3_actual_domain.txt")  # draw Figure 3(b)
    # Our data:
    # draw3("approximation_domain_sigmoid.txt", "actual_domain_sigmoid.txt")