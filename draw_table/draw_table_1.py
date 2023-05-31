
exp_num = [8, 6, 8, 6, 6, 8, 8, 6, 8, 6, 6, 8, 8,6,6,6,8,8,8,6,6,6,8,8]

def draw(write_file, read_file):
    with open(read_file, "r") as f:
        lines = f.readlines()
    times = []
    for line in lines:
        if "Time:" in line:
            ls = line.split(" ")
            for i, s in enumerate(ls):
                if s == "aver_verify_time":
                    times.append(float(ls[i+2][:6]))
    times_1 = []
    index = 0
    for num in exp_num:
        t = 0
        for i in range(index, index + num):
            t += times[i]
        times_1.append(t)
        index = index + num
    mnist_fc = sum(times_1[0:3]) + sum(times_1[6:9])
    mnist_conv = sum(times_1[3:6]) + sum(times_1[9:12])
    cifar_fc = sum(times_1[12:15]) + sum(times_1[18:21])
    cifar_conv = sum(times_1[15:18]) + sum(times_1[21:24])
    
    other_time = [2.30, 14.39, 2.25, 3.30, 4.46, 34.16, 0.88, 6.46]

    write_file.write(f"MNIST,FC,{round(mnist_fc/42,2)}s,{other_time[0]}s,{other_time[1]}s\n")
    write_file.write(f"MNIST,FC,{round(mnist_conv/42,2)}s,{other_time[2]}s,{other_time[3]}s\n")
    write_file.write(f"MNIST,FC,{round(cifar_fc/42,2)}s,{other_time[4]}s,{other_time[5]}s\n")
    write_file.write(f"MNIST,FC,{round(cifar_conv/42,2)}s,{other_time[6]}s,{other_time[7]}s\n")




if __name__ == "__main__":
    write_file = open("table_1.csv", "w")
    write_file.write("Dataset,Model,DualApp,abCrown,ERAN\n")
    draw(write_file, "../logs/run_figure_8.log")
    write_file.close()