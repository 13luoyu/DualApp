import numpy as np
from cnn_bounds_core import run_certified_bounds_core
from cnn_bounds import run_certified_bounds
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


if __name__ == "__main__":
    print("This is a demo for evaluation.")
    path_prefix = "pretrained_model/table1/"


    # Mnist CNN4-5
    # run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_sigmoid', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, cnn_cert_model=True, mnist=True)

    # Cifar-10 FNN 5*100
    # run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, cifar=True)

    last_cnn_bounds_full_core_with_LP = ""
    last_cnn_bounds_full_with_LP = ""
    for file in os.listdir("logs"):
        if "cnn_bounds_full_core_with_LP" in file:
            if file > last_cnn_bounds_full_core_with_LP:
                last_cnn_bounds_full_core_with_LP = file
        elif "cnn_bounds_full_with_LP" in file:
            if file > last_cnn_bounds_full_with_LP:
                last_cnn_bounds_full_with_LP = file
    write_file = open("demo_output.csv", "w")
    write_file.write("Dataset,Model,Nodes,DA bounds,DA time (s)\n")
    read_file = open("logs/" + last_cnn_bounds_full_core_with_LP, "r")
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
                name = names[-1][:-3]
                write_file.write(f"FNN {name},")
            write_file.write(f"{node[name]},")
        elif "[L0]" in line:
            if l0_num == 0:
                ss = line.split(" ")
                robustness = ""
                time = ""
                for i, s in enumerate(ss):
                    if s == "robustness":
                        robustness = ss[i+2][:-1]
                    elif s == "runtime" and time == "":
                        time = ss[i+2][:-1]
                l0_num = 1
                write_file.write(f"{robustness},{time}\n")

    print(last_cnn_bounds_full_with_LP)
    print(last_cnn_bounds_full_core_with_LP)