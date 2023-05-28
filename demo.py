import numpy as np
from cnn_bounds_core import run_certified_bounds_core
from cnn_bounds import run_certified_bounds


if __name__ == "__main__":
    print("This is a demo for evaluation.")
    path_prefix = "pretrained_model/table1/"


    # Mnist CNN4-5
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_sigmoid', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, cnn_cert_model=True, mnist=True)

    # Cifar-10 FNN 5*100
    run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, cifar=True)
