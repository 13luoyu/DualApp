import os

node = {
    "4-5":"8690",
    "5-5":"10690",
    "6-5":"12300",
    "8-5":"14570",
    "5x100":"510",
    "6x200":"1210",
    "1x50":"60",
    "3-2":"2514",
    "3x700":"2110",
}

def draw_demo_table(write_file, read_file):
    read_file = open("../logs/" + read_file, "r")
    l0_num = 0
    tool = 0  # 0 DA 1 NW 2 DC 3 VN 4 RV
    time_sum = 0
    DA_bound = 0
    DA_time = 0
    for line in read_file.readlines():
        if "model name" in line and tool == 0:
            model_name = line.split("/")[-1]
            if "fashion_mnist" in model_name:
                write_file.write(f"Fashion Mnist,")
            elif "cifar" in model_name:
                write_file.write(f"Cifar-10,")
            else:
                write_file.write(f"Mnist,")
            if "cnn" in model_name:
                names = model_name.split("_")
                if "fashion_mnist" in model_name:
                    name = f"{names[3][0]}-{names[4]}"
                else:
                    name = f"{names[2][0]}-{names[3]}"
                write_file.write(f"CNN {name},")
            else:
                names = model_name.split("_")
                if "fashion_mnist" in model_name:
                    name = names[3]
                    if "h5" in name:
                        name = name.split(".")[0]
                else:
                    name = names[-1][:-4]
                write_file.write(f"FNN {name},")
            write_file.write(f"{node[name]},")
        elif "[L0]" in line:
            if l0_num == 0:
                ss = line.split(" ")
                robustness = ""
                runtime = 0
                for i, s in enumerate(ss):
                    if s == "robustness":
                        robustness = ss[i+2][:-1]
                    elif s == "runtime":
                        runtime += float(ss[i+2][:-1])
                if tool == 0:
                    DA_bound = float(robustness)
                    DA_time = float(runtime)
                    write_file.write(f"{robustness},")
                else:
                    impr = round((DA_bound - float(robustness)) / float(robustness) * 100, 2)
                    write_file.write(f"{robustness},{impr},")
                    time_sum += runtime
                if tool == 4:
                    write_file.write(f"{round(DA_time,2)},{round(time_sum/4.0, 2)}\n")
                    time_sum = 0
                tool = (tool + 1) % 5
                l0_num = 1
            elif l0_num == 1:
                l0_num = 0
    read_file.close()


if __name__ == "__main__":
    write_file = open("table_1.csv", "w")
    write_file.write("Dataset,Model,Nodes,DA Bounds,NW Bounds,NW Impr. (%),DC Bounds,DC Impr. (%),VN Bounds,VN Impr. (%),RV Bounds,RV Impr. (%),DA Time (s), Others Time (s)\n")
    draw_demo_table(write_file, "../logs/run_table_1.log")
    write_file.close()