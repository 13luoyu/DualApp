import numpy as np
from cnn_bounds_core import run_verified_robustness_ratio_core
from cnn_bounds import run_verified_robustness_ratio


def run_mnist_sigmoid():
    print("Generating results for sigmoid models on MNIST")
    path_prefix = "pretrained_model/fig8/mnist/"
    # FNN
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.005)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.01)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.015)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.02)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.025)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.03)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.035)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.04)


    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__PGDK_w_0.1_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.01)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__PGDK_w_0.1_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.02)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__PGDK_w_0.1_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.03)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__PGDK_w_0.1_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.04)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__PGDK_w_0.1_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.05)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__PGDK_w_0.1_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.06)

    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__PGDK_w_0.3_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.03)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__PGDK_w_0.3_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.04)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__PGDK_w_0.3_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.05)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__PGDK_w_0.3_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.06)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__PGDK_w_0.3_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.07)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__PGDK_w_0.3_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.08)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__PGDK_w_0.3_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.09)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__PGDK_w_0.3_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.1)

    # CNN
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__Point.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.02)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__Point.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.04)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__Point.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.06)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__Point.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.08)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__Point.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.1)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__Point.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.12)

    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.1.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.06)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.1.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.08)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.1.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.1)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.1.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.12)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.1.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.14)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.1.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.16)

    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.3.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.1)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.3.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.12)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.3.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.14)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.3.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.16)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.3.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.18)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.3.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.2)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.3.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.22)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.3.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", mnist=True, eps=0.24)



def run_mnist_tanh():
    print("Generating results for tanh models on MNIST")
    path_prefix = "pretrained_model/fig8/mnist/"
    # FNN
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.005)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.01)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.015)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.02)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.025)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.03)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.035)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.04)

    run_verified_robustness_ratio(path_prefix + "ffnnTANH__PGDK_w_0.1_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.01)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__PGDK_w_0.1_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.02)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__PGDK_w_0.1_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.03)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__PGDK_w_0.1_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.04)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__PGDK_w_0.1_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.05)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__PGDK_w_0.1_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.06)

    run_verified_robustness_ratio(path_prefix + "ffnnTANH__PGDK_w_0.3_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.03)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__PGDK_w_0.3_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.04)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__PGDK_w_0.3_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.05)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__PGDK_w_0.3_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.06)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__PGDK_w_0.3_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.07)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__PGDK_w_0.3_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.08)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__PGDK_w_0.3_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.09)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__PGDK_w_0.3_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.1)

    # CNN
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__Point.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.02)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__Point.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.04)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__Point.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.06)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__Point.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.08)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__Point.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.1)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__Point.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.12)

    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.1.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.06)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.1.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.08)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.1.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.1)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.1.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.12)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.1.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.14)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.1.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.16)

    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.3.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.1)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.3.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.12)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.3.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.14)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.3.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.16)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.3.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.18)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.3.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.2)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.3.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.22)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.3.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", mnist=True, eps=0.24)



def run_cifar_sigmoid():
    print("Generating results for sigmoid models on CIFAR10")
    path_prefix = "pretrained_model/fig8/cifar10/"
    # FNN
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.0005)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.001)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.0015)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.002)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.0025)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.003)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.0035)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.004)

    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__PGDK_w_0.0078_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.001)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__PGDK_w_0.0078_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.002)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__PGDK_w_0.0078_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.003)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__PGDK_w_0.0078_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.004)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__PGDK_w_0.0078_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.005)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__PGDK_w_0.0078_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.006)

    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__PGDK_w_0.0313_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.002)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__PGDK_w_0.0313_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.004)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__PGDK_w_0.0313_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.006)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__PGDK_w_0.0313_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.008)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__PGDK_w_0.0313_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.01)
    run_verified_robustness_ratio(path_prefix + "ffnnSIGMOID__PGDK_w_0.0313_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.012)

    # CNN
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__Point.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.002)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__Point.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.004)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__Point.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.006)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__Point.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.008)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__Point.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.01)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__Point.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.012)

    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.0078.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.002)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.0078.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.004)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.0078.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.006)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.0078.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.008)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.0078.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.01)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.0078.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.012)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.0078.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.014)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.0078.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.016)

    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.0313.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.002)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.0313.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.004)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.0313.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.006)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.0313.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.008)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.0313.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.01)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.0313.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.012)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.0313.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.014)
    run_verified_robustness_ratio_core(path_prefix + "convMedGSIGMOID__PGDK_w_0.0313.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="sigmoid", cifar=True, eps=0.016)



def run_cifar_tanh():
    print("Generating results for tanh models on CIFAR10")
    path_prefix = "pretrained_model/fig8/cifar10/"
    # FNN
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.0005)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.001)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.0015)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.002)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.0025)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.003)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.0035)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__Point_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.004)

    run_verified_robustness_ratio(path_prefix + "ffnnTANH__PGDK_w_0.0078_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.001)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__PGDK_w_0.0078_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.002)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__PGDK_w_0.0078_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.003)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__PGDK_w_0.0078_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.004)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__PGDK_w_0.0078_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.005)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__PGDK_w_0.0078_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.006)

    run_verified_robustness_ratio(path_prefix + "ffnnTANH__PGDK_w_0.0313_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.002)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__PGDK_w_0.0313_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.004)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__PGDK_w_0.0313_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.006)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__PGDK_w_0.0313_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.008)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__PGDK_w_0.0313_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.01)
    run_verified_robustness_ratio(path_prefix + "ffnnTANH__PGDK_w_0.0313_6_500.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.012)

    # CNN
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__Point.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.002)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__Point.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.004)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__Point.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.006)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__Point.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.008)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__Point.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.01)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__Point.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.012)

    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.0078.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.002)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.0078.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.004)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.0078.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.006)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.0078.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.008)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.0078.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.01)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.0078.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.012)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.0078.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.014)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.0078.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.016)

    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.0313.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.002)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.0313.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.004)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.0313.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.006)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.0313.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.008)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.0313.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.01)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.0313.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.012)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.0313.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.014)
    run_verified_robustness_ratio_core(path_prefix + "convMedGTANH__PGDK_w_0.0313.h5", 100, 105, 1, method="guided_by_endpoint", sample_num=1000, eran_fnn=True, activation="tanh", cifar=True, eps=0.016)




if __name__ == "__main__":
    run_mnist_sigmoid()
    run_mnist_tanh()
    run_cifar_sigmoid()
    run_cifar_tanh()