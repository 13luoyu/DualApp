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
    for line in read_file.readlines():
        if "model name" in line:
            model_name = line.split("/")[-1]
            if "mnist" in model_name:
                write_file.write(f"Mnist,")
            elif "cifar" in model_name:
                write_file.write(f"Cifar-10,")
            else:
                write_file.write(f"Fashion Mnist,")
            if "cnn" in model_name:
                names = model_name.split("_")
                name = f"{names[2][0]}-{names[3]}"
                write_file.write(f"CNN {name},")
            else:
                names = model_name.split("_")
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
                l0_num = 1
                write_file.write(f"{robustness},{runtime}\n")
            elif l0_num == 1:
                l0_num = 0
    read_file.close()


if __name__ == "__main__":
    last_cnn_bounds_full_core_with_LP = ""
    last_cnn_bounds_full_with_LP = ""
    for file in os.listdir("../logs"):
        if "cnn_bounds_full_core_with_LP" in file:
            if file > last_cnn_bounds_full_core_with_LP:
                last_cnn_bounds_full_core_with_LP = file
        elif "cnn_bounds_full_with_LP" in file:
            if file > last_cnn_bounds_full_with_LP:
                last_cnn_bounds_full_with_LP = file
    write_file = open("demo_table.csv", "w")
    write_file.write("Dataset,Model,Nodes,DA bounds,DA time (s)\n")
    draw_demo_table(write_file, last_cnn_bounds_full_core_with_LP)
    draw_demo_table(write_file, last_cnn_bounds_full_with_LP)
    write_file.close()