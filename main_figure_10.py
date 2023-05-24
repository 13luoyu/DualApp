import numpy as np
from cnn_bounds_core import run_certified_bounds_core
from cnn_bounds import run_certified_bounds



if __name__ == "__main__":
    path_prefix = "pretrained_model/fig9_10/"
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_1_5_sigmoid_local.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_2_5_sigmoid_local.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_3_5_sigmoid_local.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_4_5_sigmoid_local.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, mnist=True)

    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_1_5_sigmoid_local.h5', 100, 105, 1, method='gradient_descent', mnist=True, step_tmp = 0.50)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_2_5_sigmoid_local.h5', 100, 105, 1, method='gradient_descent', mnist=True, step_tmp = 0.45)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_3_5_sigmoid_local.h5', 100, 105, 1, method='gradient_descent', mnist=True, step_tmp = 0.40)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_2layer_4_5_sigmoid_local.h5', 100, 105, 1, method='gradient_descent', mnist=True, step_tmp = 0.40)

    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_1_5_sigmoid_local.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_2_5_sigmoid_local.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_3_5_sigmoid_local.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_4_5_sigmoid_local.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, fashion_mnist=True)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_1_5_sigmoid_local.h5', 100, 105, 1, method='gradient_descent', fashion_mnist=True, step_tmp = 0.45)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_2_5_sigmoid_local.h5', 100, 105, 1, method='gradient_descent', fashion_mnist=True, step_tmp = 0.45)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_3_5_sigmoid_local.h5', 100, 105, 1, method='gradient_descent', fashion_mnist=True, step_tmp = 0.40)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_2layer_4_5_sigmoid_local.h5', 100, 105, 1, method='gradient_descent', fashion_mnist=True, step_tmp = 0.45)

    run_certified_bounds(path_prefix + 'mnist_fnn_1x100_sigmoid_local.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, mnist=True)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x150_sigmoid_local.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, mnist=True)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x200_sigmoid_local.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, mnist=True)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x250_sigmoid_local.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, mnist=True)

    run_certified_bounds(path_prefix + 'mnist_fnn_1x100_sigmoid_local.h5', 100, 105, 1, method='gradient_descent', mnist=True, step_tmp = 0.45)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x150_sigmoid_local.h5', 100, 105, 1, method='gradient_descent', mnist=True, step_tmp = 0.45)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x200_sigmoid_local.h5', 100, 105, 1, method='gradient_descent', mnist=True, step_tmp = 0.45)
    run_certified_bounds(path_prefix + 'mnist_fnn_1x250_sigmoid_local.h5', 100, 105, 1, method='gradient_descent', mnist=True, step_tmp = 0.45)

    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x100_sigmoid_local.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, fashion_mnist=True)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x150_sigmoid_local.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, fashion_mnist=True)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x200_sigmoid_local.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, fashion_mnist=True)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x250_sigmoid_local.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, fashion_mnist=True)

    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x100_sigmoid_local.h5', 100, 105, 1, method='gradient_descent', fashion_mnist=True, step_tmp = 0.45)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x150_sigmoid_local.h5', 100, 105, 1, method='gradient_descent', fashion_mnist=True, step_tmp = 0.45)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x200_sigmoid_local.h5', 100, 105, 1, method='gradient_descent', fashion_mnist=True, step_tmp = 0.45)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x250_sigmoid_local.h5', 100, 105, 1, method='gradient_descent', fashion_mnist=True, step_tmp = 0.45)
