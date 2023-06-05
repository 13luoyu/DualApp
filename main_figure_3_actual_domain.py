import numpy as np
from scipy.optimize import minimize

def printlog(s):
    print(s, file=open("logs/figure_3_actual_domain.txt", "a"))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# def y1_(args):
#     y1 = lambda x: args[8] / (1.0 + np.exp(-(args[4] / (1.0 + np.exp(-(args[0] * x[0] + args[2] * x[1])))) + args[6] / (1 + np.exp(-(args[1] * x[0] + args[3] * x[1]))))) + \
#         args[10] / (1.0 + np.exp(-(args[5] / (1.0 + np.exp(-(args[0] * x[0] + args[2] * x[1]))) + args[7] / (1.0 + np.exp(-(args[1] * x[0] + args[3] * x[1]))))))
#     return y1

# def y2_(args):
#     y2 = lambda x: args[9] / (1.0 + np.exp(-(args[4] / (1.0 + np.exp(-(args[0] * x[0] + args[2] * x[1])))) + args[6] / (1 + np.exp(-(args[1] * x[0] + args[3] * x[1]))))) + \
#         args[11] / (1.0 + np.exp(-(args[5] / (1.0 + np.exp(-(args[0] * x[0] + args[2] * x[1]))) + args[7] / (1.0 + np.exp(-(args[1] * x[0] + args[3] * x[1]))))))
#     return y2

def x3_min(args):
    x3 = lambda x: sigmoid(args[0] * x[0] + args[2] * x[1])
    return x3
def x3_max(args):
    x3 = lambda x: -sigmoid(args[0] * x[0] + args[2] * x[1])
    return x3
def x4_min(args):
    x4 = lambda x: sigmoid(args[1] * x[0] + args[3] * x[1])
    return x4
def x4_max(args):
    x4 = lambda x: -sigmoid(args[1] * x[0] + args[3] * x[1])
    return x4
def x5_min(args):
    x5 = lambda x: sigmoid(args[4] * sigmoid(args[0] * x[0] + args[2] * x[1]) + args[6] * sigmoid(args[1] * x[0] + args[3] * x[1]))
    return x5
def x5_max(args):
    x5 = lambda x: -sigmoid(args[4] * sigmoid(args[0] * x[0] + args[2] * x[1]) + args[6] * sigmoid(args[1] * x[0] + args[3] * x[1]))
    return x5
def x6_min(args):
    x6 = lambda x: sigmoid(args[5] * sigmoid(args[0] * x[0] + args[2] * x[1]) + args[7] * sigmoid(args[1] * x[0] + args[3] * x[1]))
    return x6
def x6_max(args):
    x6 = lambda x: -sigmoid(args[5] * sigmoid(args[0] * x[0] + args[2] * x[1]) + args[7] * sigmoid(args[1] * x[0] + args[3] * x[1]))
    return x6

def x3_min_2(args):
    x3 = lambda x: tanh(args[0] * x[0] + args[2] * x[1])
    return x3
def x3_max_2(args):
    x3 = lambda x: -tanh(args[0] * x[0] + args[2] * x[1])
    return x3
def x4_min_2(args):
    x4 = lambda x: tanh(args[1] * x[0] + args[3] * x[1])
    return x4
def x4_max_2(args):
    x4 = lambda x: -tanh(args[1] * x[0] + args[3] * x[1])
    return x4
def x5_min_2(args):
    x5 = lambda x: tanh(args[4] * tanh(args[0] * x[0] + args[2] * x[1]) + args[6] * tanh(args[1] * x[0] + args[3] * x[1]))
    return x5
def x5_max_2(args):
    x5 = lambda x: -tanh(args[4] * tanh(args[0] * x[0] + args[2] * x[1]) + args[6] * tanh(args[1] * x[0] + args[3] * x[1]))
    return x5
def x6_min_2(args):
    x6 = lambda x: tanh(args[5] * tanh(args[0] * x[0] + args[2] * x[1]) + args[7] * tanh(args[1] * x[0] + args[3] * x[1]))
    return x6
def x6_max_2(args):
    x6 = lambda x: -tanh(args[5] * tanh(args[0] * x[0] + args[2] * x[1]) + args[7] * tanh(args[1] * x[0] + args[3] * x[1]))
    return x6

def y1_min(args):
    y1 = lambda x: args[8] * sigmoid(args[4] * sigmoid(args[0] * x[0] + args[2] * x[1]) + args[6] * sigmoid(args[1] * x[0] + args[3] * x[1])) + \
        args[10] * sigmoid(args[5] * sigmoid(args[0] * x[0] + args[2] * x[1]) + args[7] * sigmoid(args[1] * x[0] + args[3] * x[1]))
    return y1

def y1_max(args):
    y1 = lambda x: - (args[8] * sigmoid(args[4] * sigmoid(args[0] * x[0] + args[2] * x[1]) + args[6] * sigmoid(args[1] * x[0] + args[3] * x[1])) + \
        args[10] * sigmoid(args[5] * sigmoid(args[0] * x[0] + args[2] * x[1]) + args[7] * sigmoid(args[1] * x[0] + args[3] * x[1])))
    return y1

def y2_min(args):
    y2 = lambda x: args[9] * sigmoid(args[4] * sigmoid(args[0] * x[0] + args[2] * x[1]) + args[6] * sigmoid(args[1] * x[0] + args[3] * x[1])) + \
        args[11] * sigmoid(args[5] * sigmoid(args[0] * x[0] + args[2] * x[1]) + args[7] * sigmoid(args[1] * x[0] + args[3] * x[1]))
    return y2

def y2_max(args):
    y2 = lambda x: - (args[9] * sigmoid(args[4] * sigmoid(args[0] * x[0] + args[2] * x[1]) + args[6] * sigmoid(args[1] * x[0] + args[3] * x[1])) + \
        args[11] * sigmoid(args[5] * sigmoid(args[0] * x[0] + args[2] * x[1]) + args[7] * sigmoid(args[1] * x[0] + args[3] * x[1])))
    return y2

def y1_min_2(args):
    y1 = lambda x: args[8] * tanh(args[4] * tanh(args[0] * x[0] + args[2] * x[1]) + args[6] * tanh(args[1] * x[0] + args[3] * x[1])) + \
        args[10] * tanh(args[5] * tanh(args[0] * x[0] + args[2] * x[1]) + args[7] * tanh(args[1] * x[0] + args[3] * x[1]))
    return y1

def y1_max_2(args):
    y1 = lambda x: - (args[8] * tanh(args[4] * tanh(args[0] * x[0] + args[2] * x[1]) + args[6] * tanh(args[1] * x[0] + args[3] * x[1])) + \
        args[10] * tanh(args[5] * tanh(args[0] * x[0] + args[2] * x[1]) + args[7] * tanh(args[1] * x[0] + args[3] * x[1])))
    return y1

def y2_min_2(args):
    y2 = lambda x: args[9] * tanh(args[4] * tanh(args[0] * x[0] + args[2] * x[1]) + args[6] * tanh(args[1] * x[0] + args[3] * x[1])) + \
        args[11] * tanh(args[5] * tanh(args[0] * x[0] + args[2] * x[1]) + args[7] * tanh(args[1] * x[0] + args[3] * x[1]))
    return y2

def y2_max_2(args):
    y2 = lambda x: - (args[9] * tanh(args[4] * tanh(args[0] * x[0] + args[2] * x[1]) + args[6] * tanh(args[1] * x[0] + args[3] * x[1])) + \
        args[11] * tanh(args[5] * tanh(args[0] * x[0] + args[2] * x[1]) + args[7] * tanh(args[1] * x[0] + args[3] * x[1])))
    return y2

def con(args):
    cons = ({'type':'ineq', 'fun':lambda x: x[0] - args[0]},\
        {'type':'ineq', 'fun':lambda x: -x[0] + args[1]},\
        {'type':'ineq', 'fun':lambda x: x[1] - args[2]},\
        {'type':'ineq', 'fun':lambda x: -x[1] + args[3]})
    return cons


def deal_with_sigmoid(file_name):
    with open(file_name) as f:
        lines = f.readlines()
    y_args = []
    con_args = []
    step = 0
    for line in lines:
        # 读权重
        if step == 0:
            if "Weights:" in line:
                step = 1
        elif step == 1 and "NeWise" not in line:
            y_args += line.strip().split(" ")
        elif step == 1 and "NeWise" in line:
            step = 2
        elif step == 2:
            l = line[4:-2]
            con_args += l.strip().split(" ")

        # y_args = (1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 3.0, 2.0, 1.0, -2.0, -1.0, 1.0)
        # con_args = (-1.0, 1.0, -1.0, 1.0)
        if len(y_args) == 12 and len(con_args) == 4:
            y_args = [float(a) for a in y_args]
            con_args = [float(a) for a in con_args]
            # print(y_args, con_args)
            cons = con(con_args)
            x0 = np.asarray((0.5, 0.5))
            # x3最小值，最大值
            x31, x32 = minimize(x3_min(y_args), x0, method="SLSQP", constraints=cons), minimize(x3_max(y_args), x0, method="SLSQP", constraints=cons)
            # x4最小值，最大值
            x41, x42 = minimize(x4_min(y_args), x0, method="SLSQP", constraints=cons), minimize(x4_max(y_args), x0, method="SLSQP", constraints=cons)
            # x5最小值，最大值
            x51, x52 = minimize(x5_min(y_args), x0, method="SLSQP", constraints=cons), minimize(x5_max(y_args), x0, method="SLSQP", constraints=cons)
            # x6最小值，最大值
            x61, x62 = minimize(x6_min(y_args), x0, method="SLSQP", constraints=cons), minimize(x6_max(y_args), x0, method="SLSQP", constraints=cons)
            # y1最小值，最大值
            y11, y12 = minimize(y1_min(y_args), x0, method="SLSQP", constraints=cons), minimize(y1_max(y_args), x0, method="SLSQP", constraints=cons)
            # y2最小值，最大值
            y21, y22 = minimize(y2_min(y_args), x0, method="SLSQP", constraints=cons), minimize(y2_max(y_args), x0, method="SLSQP", constraints=cons)
            printlog("{:.3f} {:.3f}".format(con_args[0], con_args[1]))
            printlog("{:.3f} {:.3f}".format(con_args[2], con_args[3]))
            printlog("{:.3f} {:.3f}".format(x31.fun, -x32.fun))
            printlog("{:.3f} {:.3f}".format(x41.fun, -x42.fun))
            printlog("{:.3f} {:.3f}".format(x51.fun, -x52.fun))
            printlog("{:.3f} {:.3f}".format(x61.fun, -x62.fun))
            printlog("{:.3f} {:.3f}".format(y11.fun, -y12.fun))
            printlog("{:.3f} {:.3f}".format(y21.fun, -y22.fun))
            printlog()
            y_args = []
            con_args = []
            step = 0



def deal_with_tanh(file_name):
    with open(file_name) as f:
        lines = f.readlines()
    y_args = []
    con_args = []
    step = 0
    for line in lines:
        # 读权重
        if step == 0:
            if "Weights:" in line:
                step = 1
        elif step == 1 and "NeWise" not in line:
            y_args += line.strip().split(" ")
        elif step == 1 and "NeWise" in line:
            step = 2
        elif step == 2:
            l = line[4:-2]
            con_args += l.strip().split(" ")

        # y_args = (1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 3.0, 2.0, 1.0, -2.0, -1.0, 1.0)
        # con_args = (-1.0, 1.0, -1.0, 1.0)
        if len(y_args) == 12 and len(con_args) == 4:
            y_args = [float(a) for a in y_args]
            con_args = [float(a) for a in con_args]
            # print(y_args, con_args)
            cons = con(con_args)
            x0 = np.asarray((0.5, 0.5))
            # x3最小值，最大值
            x31, x32 = minimize(x3_min_2(y_args), x0, method="SLSQP", constraints=cons), minimize(x3_max_2(y_args), x0, method="SLSQP", constraints=cons)
            # x4最小值，最大值
            x41, x42 = minimize(x4_min_2(y_args), x0, method="SLSQP", constraints=cons), minimize(x4_max_2(y_args), x0, method="SLSQP", constraints=cons)
            # x5最小值，最大值
            x51, x52 = minimize(x5_min_2(y_args), x0, method="SLSQP", constraints=cons), minimize(x5_max_2(y_args), x0, method="SLSQP", constraints=cons)
            # x6最小值，最大值
            x61, x62 = minimize(x6_min_2(y_args), x0, method="SLSQP", constraints=cons), minimize(x6_max_2(y_args), x0, method="SLSQP", constraints=cons)
            # y1最小值，最大值
            y11, y12 = minimize(y1_min_2(y_args), x0, method="SLSQP", constraints=cons), minimize(y1_max_2(y_args), x0, method="SLSQP", constraints=cons)
            # y2最小值，最大值
            y21, y22 = minimize(y2_min_2(y_args), x0, method="SLSQP", constraints=cons), minimize(y2_max_2(y_args), x0, method="SLSQP", constraints=cons)
            printlog("{:.3f} {:.3f}".format(con_args[0], con_args[1]))
            printlog("{:.3f} {:.3f}".format(con_args[2], con_args[3]))
            printlog("{:.3f} {:.3f}".format(x31.fun, -x32.fun))
            printlog("{:.3f} {:.3f}".format(x41.fun, -x42.fun))
            printlog("{:.3f} {:.3f}".format(x51.fun, -x52.fun))
            printlog("{:.3f} {:.3f}".format(x61.fun, -x62.fun))
            printlog("{:.3f} {:.3f}".format(y11.fun, -y12.fun))
            printlog("{:.3f} {:.3f}".format(y21.fun, -y22.fun))
            printlog()
            y_args = []
            con_args = []
            step = 0



if __name__ == '__main__':
    deal_with_sigmoid("table_one_20230213_175411.txt")
    # deal_with_tanh("table_one_20230213_175924.txt")
