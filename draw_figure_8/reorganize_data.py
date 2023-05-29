

if __name__ == "__main__":
    with open("../logs/run_figure_8.log", "r") as f:
        lines = f.readlines()
    
filename_index = ["mnist_fc_normal.txt", "mnist_fc_pgd1.txt", "mnist_fc_pgd2.txt", "mnist_conv_normal.txt", "mnist_conv_pgd1.txt", "mnist_conv_pgd2.txt", "cifar_fc_normal.txt", "cifar_fc_pgd1.txt", "cifar_fc_pgd2.txt", "cifar_conv_normal.txt", "cifar_conv_pgd1.txt", "cifar_conv_pgd2.txt"]

if __name__ == "__main__":
    with open("../logs/run_figure_8.log", "r") as f:
        lines = f.readlines()
    robust_ratio, time, eps = [], [], []
    for line in lines:
        if "### Summary" in line:
            ss = line.split(" ")
            for i, s in enumerate(ss):
                if "eps" in s:
                    eps_i = float(s[4:-1])
                    eps.append(eps_i)
                if "robust_images" in s:
                    robust_rate = int(s[14:-1])
                    robust_ratio.append(robust_rate)
                    break
        elif "Time" in line:
            ss = line.split(" ")
            runtime = 0.0
            for i, s in enumerate(ss):
                if "time" in s:
                    t_time = ss[i+2][:-1]
                    if t_time == "0.0":
                        continue
                    else:
                        t_time = round(float(t_time[:6]), 2)
                    runtime += t_time
            time.append(runtime)
    begin, end = [0], []
    last_eps_i = 0
    for end_i, eps_i in enumerate(eps):
        if eps_i < last_eps_i:
            end.append(end_i)
            begin.append(end_i)
        last_eps_i = eps_i
    end.append(len(eps))
    for i, filename in enumerate(filename_index):
        # 一个filename对应2个begin，分别为sigmoid和tanh
        # 前一半filename对应两个begin分别在i和i+len(begin)/4的位置
        # 后一半filename对应两个begin分别在i+len(begin)/4和i+len(begin)/2的位置
        with open(filename, "r") as f:
            lines = f.readlines()
        eps_want = []
        for line in lines:
            if line == "\n":
                break
            eps_want.append(float(line[:-1]))
        acc1, acc2 = [], []
        if i < len(filename_index)/2:
            begin_i, end_i = begin[i], end[i]
        else:
            begin_i, end_i = begin[int(i+len(begin)/4)], end[int(i+len(begin)/4)]
        for j, eps_j in enumerate(eps):
            if j < begin_i or j >= end_i:
                continue
            if eps_j in eps_want:
                acc1.append(robust_ratio[j])
        if i < len(filename_index)/2:
            begin_i, end_i = begin[int(i+len(begin)/4)], end[int(i+len(begin)/4)]
        else:
            begin_i, end_i = begin[int(i+len(begin)/2)], end[int(i+len(begin)/2)]
        for j, eps_j in enumerate(eps):
            if j < begin_i or j >= end_i:
                continue
            if eps_j in eps_want:
                acc2.append(robust_ratio[j])
        # 写数据，写到文件中
        if i == 0:
            enter = 0
            index = 0
            for j, line in enumerate(lines):
                if line == "\n":
                    enter += 1
                    if enter > 2:
                        break
                    index = 0
                    continue
                if enter == 1:
                    lines[j] = str(acc1[index]) + "\n"
                    index += 1
                if enter == 2:
                    lines[j] = str(acc2[index]) + "\n"
                    index += 1
        else:
            enter = 0
            index = 0
            for j, line in enumerate(lines):
                if line == "\n":
                    enter += 1
                    if enter > 4:
                        break
                    index = 0
                    continue
                if enter == 1:
                    lines[j] = str(acc1[index]) + "\n"
                    index += 1
                if enter == 4:
                    lines[j] = str(acc2[index]) + "\n"
                    index += 1
        with open(filename, "w") as f:
            f.write("".join(lines))