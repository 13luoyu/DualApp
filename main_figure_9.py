import numpy as np
from cnn_bounds import run_certified_bounds


if __name__ == "__main__":
    path_prefix = "pretrained_model/fig9_10/"
    for sample_num in range(100, 2001, 100):
        run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x150_sigmoid_local.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=sample_num, fashion_mnist=True)
    step_range = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
    for step_tmp in step_range:
        run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x150_sigmoid_local.h5', 100, 105, 1, method='gradient_descent', fashion_mnist=True, step_tmp = step_tmp)