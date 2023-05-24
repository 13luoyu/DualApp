import numpy as np
from cnn_bounds_core import run_certified_bounds_core
from cnn_bounds import run_certified_bounds




if __name__ == "__main__":

    print("generating results in Table 1")
    path_prefix = "pretrained_model/table1/"
    # Mnist
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_sigmoid', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, cnn_cert_model=True, mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_sigmoid', 100, 105, 1, method='NeWise', cnn_cert_model=True, mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_sigmoid', 100, 105, 1, method='DeepCert', cnn_cert_model=True, mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_sigmoid', 100, 105, 1, method='VeriNet', cnn_cert_model=True, mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_4layer_5_3_sigmoid', 100, 105, 1, method='RobustVerifier', cnn_cert_model=True, mnist=True)

    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3.h5', 100, 105, 1, method='NeWise', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3.h5', 100, 105, 1, method='DeepCert', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3.h5', 100, 105, 1, method='VeriNet', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_5layer_5_3.h5', 100, 105, 1, method='RobustVerifier', mnist=True)

    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3.h5', 100, 105, 1, method='NeWise', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3.h5', 100, 105, 1, method='DeepCert', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3.h5', 100, 105, 1, method='VeriNet', mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_6layer_5_3.h5', 100, 105, 1, method='RobustVerifier', mnist=True)

    run_certified_bounds_core(path_prefix + 'mnist_cnn_8layer_5_3_sigmoid', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, cnn_cert_model=True, mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_8layer_5_3_sigmoid', 100, 105, 1, method='NeWise', cnn_cert_model=True, mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_8layer_5_3_sigmoid', 100, 105, 1, method='DeepCert', cnn_cert_model=True, mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_8layer_5_3_sigmoid', 100, 105, 1, method='VeriNet', cnn_cert_model=True, mnist=True)
    run_certified_bounds_core(path_prefix + 'mnist_cnn_8layer_5_3_sigmoid', 100, 105, 1, method='RobustVerifier', cnn_cert_model=True, mnist=True)

    run_certified_bounds(path_prefix + 'mnist_ffnn_5x100.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, mnist=True)
    run_certified_bounds(path_prefix + 'mnist_ffnn_5x100.h5', 100, 105, 1, method='NeWise', mnist=True)
    run_certified_bounds(path_prefix + 'mnist_ffnn_5x100.h5', 100, 105, 1, method='DeepCert', mnist=True)
    run_certified_bounds(path_prefix + 'mnist_ffnn_5x100.h5', 100, 105, 1, method='VeriNet', mnist=True)
    run_certified_bounds(path_prefix + 'mnist_ffnn_5x100.h5', 100, 105, 1, method='RobustVerifier', mnist=True)

    run_certified_bounds(path_prefix + 'ffnnSIGMOID__Point_6x200.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, eran_fnn=True, mnist=True)
    run_certified_bounds(path_prefix + 'ffnnSIGMOID__Point_6x200.h5', 100, 105, 1, method='NeWise', eran_fnn=True, mnist=True)
    run_certified_bounds(path_prefix + 'ffnnSIGMOID__Point_6x200.h5', 100, 105, 1, method='DeepCert', eran_fnn=True, mnist=True)
    run_certified_bounds(path_prefix + 'ffnnSIGMOID__Point_6x200.h5', 100, 105, 1, method='VeriNet', eran_fnn=True, mnist=True)
    run_certified_bounds(path_prefix + 'ffnnSIGMOID__Point_6x200.h5', 100, 105, 1, method='RobustVerifier', eran_fnn=True, mnist=True)
    
    # Fashion Mnist
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_4layer_5_3.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_4layer_5_3.h5', 100, 105, 1, method='NeWise', fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_4layer_5_3.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_4layer_5_3.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_4layer_5_3.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True)

    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_5layer_5_3.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_5layer_5_3.h5', 100, 105, 1, method='NeWise', fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_5layer_5_3.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_5layer_5_3.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_5layer_5_3.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_6layer_5_3.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_6layer_5_3.h5', 100, 105, 1, method='NeWise', fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_6layer_5_3.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_6layer_5_3.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_6layer_5_3.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True)
    
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_8layer_5_3_sigmoid_myself.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_8layer_5_3_sigmoid_myself.h5', 100, 105, 1, method='NeWise', fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_8layer_5_3_sigmoid_myself.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_8layer_5_3_sigmoid_myself.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True)
    run_certified_bounds_core(path_prefix + 'fashion_mnist_cnn_8layer_5_3_sigmoid_myself.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True)

    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x50_sigmoid_local.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, fashion_mnist=True)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x50_sigmoid_local.h5', 100, 105, 1, method='NeWise', fashion_mnist=True)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x50_sigmoid_local.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x50_sigmoid_local.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True)
    run_certified_bounds(path_prefix + 'fashion_mnist_fnn_1x50_sigmoid_local.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True)

    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_5x100.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, fashion_mnist=True)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_5x100.h5', 100, 105, 1, method='NeWise', fashion_mnist=True)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_5x100.h5', 100, 105, 1, method='DeepCert', fashion_mnist=True)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_5x100.h5', 100, 105, 1, method='VeriNet', fashion_mnist=True)
    run_certified_bounds(path_prefix + 'fashion_mnist_ffnn_5x100.h5', 100, 105, 1, method='RobustVerifier', fashion_mnist=True)

    # cifar-10
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3.h5', 100, 105, 1, method='NeWise', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3.h5', 100, 105, 1, method='DeepCert', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3.h5', 100, 105, 1, method='VeriNet', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_3layer_2_3.h5', 100, 105, 1, method='RobustVerifier', cifar=True)

    run_certified_bounds_core(path_prefix + 'cifar10_cnn_5layer_5_3.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_5layer_5_3.h5', 100, 105, 1, method='NeWise', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_5layer_5_3.h5', 100, 105, 1, method='DeepCert', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_5layer_5_3.h5', 100, 105, 1, method='VeriNet', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_5layer_5_3.h5', 100, 105, 1, method='RobustVerifier', cifar=True)
    
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_6layer_5_3.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_6layer_5_3.h5', 100, 105, 1, method='NeWise', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_6layer_5_3.h5', 100, 105, 1, method='DeepCert', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_6layer_5_3.h5', 100, 105, 1, method='VeriNet', cifar=True)
    run_certified_bounds_core(path_prefix + 'cifar10_cnn_6layer_5_3.h5', 100, 105, 1, method='RobustVerifier', cifar=True)
    
    run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, cifar=True)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100.h5', 100, 105, 1, method='NeWise', cifar=True)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100.h5', 100, 105, 1, method='DeepCert', cifar=True)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100.h5', 100, 105, 1, method='VeriNet', cifar=True)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_5x100.h5', 100, 105, 1, method='RobustVerifier', cifar=True)

    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x700.h5', 100, 105, 1, method='guided_by_endpoint', sample_num=1000, cifar=True)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x700.h5', 100, 105, 1, method='NeWise', cifar=True)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x700.h5', 100, 105, 1, method='DeepCert', cifar=True)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x700.h5', 100, 105, 1, method='VeriNet', cifar=True)
    run_certified_bounds(path_prefix + 'cifar10_ffnn_3x700.h5', 100, 105, 1, method='RobustVerifier', cifar=True)
