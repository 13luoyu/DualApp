from tracemalloc import start
from turtle import st
from cnn_bounds_core import *
import numpy as np
from activations import *
import copy
import time
from numba import njit
import matplotlib.pyplot as plt

@njit
def conv(W, x, pad, stride):
    p_hl, p_hr, p_wl, p_wr = pad
    s_h, s_w = stride
    y = np.zeros((int((x.shape[0]-W.shape[1]+p_hl+p_hr)/s_h)+1, int((x.shape[1]-W.shape[2]+p_wl+p_wr)/s_w)+1, W.shape[0]), dtype=np.float32)
    for a in range(y.shape[0]):
        for b in range(y.shape[1]):
            for c in range(y.shape[2]):
                for i in range(W.shape[1]):
                    for j in range(W.shape[2]):
                        for k in range(W.shape[3]):
                            if 0<=s_h*a+i-p_hl<x.shape[0] and 0<=s_w*b+j-p_wl<x.shape[1]:
                                y[a,b,c] += W[c,i,j,k]*x[s_h*a+i-p_hl,s_w*b+j-p_wl,k]
    return y

@njit
def under_conv_full(W, b, pad, stride, x0):
    y0 = conv(W, x0, pad, stride)
    for k in range(W.shape[0]):
        y0[:,:,k] += b[k]
        
    return y0

def forward_propagation(x0, weights, biases, shapes, pads, strides, activation):
    sample_result = [copy.deepcopy(x0)]
    y = x0

    for i in range(len(weights)):
        z = under_conv_full(weights[i], biases[i], pads[i], strides[i], y)
        sample_result.append(copy.deepcopy(z))
        if activation == 'sigmoid':
            y = sigmoid(z)
        elif activation == 'tanh':
            y = tanh(z)
        elif activation == 'atan':
            y = atan(z)
    
    return copy.deepcopy(sample_result)

def fnn_forward_propagation(x0, weights, biases, shapes, pads, strides, activation):
    sample_result = [copy.deepcopy(x0)]
    y = x0

    for i in range(len(weights)):
        if pads[i] is None:
            sample_result.append(copy.deepcopy(y))
            continue
        
        z = under_conv_full(weights[i], biases[i], pads[i], strides[i], y)
        sample_result.append(copy.deepcopy(z))
        if activation == 'sigmoid':
            y = sigmoid(z)
        elif activation == 'tanh':
            y = tanh(z)
        elif activation == 'atan':
            y = atan(z)
    
    return copy.deepcopy(sample_result)

def obtain_under_range(under_LBs, under_UBs, weights, biases, shapes, pads, strides, x0, activation='sigmoid', init=False):
    y = x0
    for i in range(len(weights)):
        z = under_conv_full(weights[i], biases[i], pads[i], strides[i], y)
        
        if init:
            under_LBs.append(copy.deepcopy(z))
            under_UBs.append(copy.deepcopy(z))
        else:
            for a in range(z.shape[0]):
                for b in range(z.shape[1]):
                    for c in range(z.shape[2]):
                        if z[a][b][c] < under_LBs[i+1][a][b][c]:
                            under_LBs[i+1][a][b][c] = z[a][b][c]
                        if z[a][b][c] > under_UBs[i+1][a][b][c]:
                            under_UBs[i+1][a][b][c] = z[a][b][c]
       
        if activation == 'sigmoid':
            y = sigmoid(z)
        elif activation == 'tanh':
            y = tanh(z)
        elif activation == 'atan':
            y = atan(z)
    
    return under_LBs, under_UBs

if __name__ == '__main__':
    # file_name = "pretrained_model/from_eran/ffnnSIGMOID__Point_6x200.h5"
    # file_name = "pretrained_model/models/models_with_positive_weights/sigmoid/mnist_ffnn_5x100_with_positive_weights.h5"
    # file_name = "pretrained_model/models/one_layer_mixed_models/mnist_cnn_2layer_1_5_sigmoid_local.h5"
    file_name = "pretrained_model/models/one_layer_mixed_models/mnist_fnn_1x50_sigmoid_local.h5"
    #file_name = "models/mnist_cnn_4layer_5_3_sigmoid"
    
    keras_model = load_model(file_name)
    model = Model(keras_model)
    
    dataset = 'mnist'
    n_samples = 100
    data_from_local = True
    cnn_cert_model = False
    vnn_comp_model = False
    eran_fnn = True
    eran_cnn = False
    activation = 'sigmoid'
    inputs, targets, true_labels, true_ids, img_info = generate_data('mnist', samples=n_samples, data_from_local=data_from_local, targeted=True, random_and_least_likely = True, target_type = 0b0010, predictor=model.model.predict, start=0, cnn_cert_model=cnn_cert_model)
        
    start_time = time.time()

    print(fnn_forward_propagation(inputs[0], model.weights, model.biases, model.shapes, model.pads, model.strides, activation))
    # 1. 产生 sample_num 个随机样本，使用 numpy.ndarray 结构存储，shape: (sample_num, input_shape)
    eps = 0.02
    sample_num = 500
    samples_shape = (sample_num, inputs[0].shape[0], inputs[0].shape[1], inputs[0].shape[2])
    samples = np.random.uniform(inputs[0] - eps, inputs[0] + eps, samples_shape).astype(np.float32)
    print(samples.dtype)
    print(samples.shape)
    
    # 2. 对这 sample_num 个随机样本进行前向传播，创建 sample_results 用于存储 sample_num 个采样点在每个节点的取值
    # 因为每一层的 shape 都不一样，所以对于每个采样点在每个节点的取值只能用 list 进行存储，
    # 进而 sample_num 个采样点的总结果也存储于 list 结构中，每个采样点在每一层的取值结果用 numpy.array 存储
    activation = 'sigmoid'
    sample_results = []
    # start_time = time.time()
    for i in range(sample_num):
        sample_results.append(fnn_forward_propagation(samples[i], model.weights, model.biases, model.shapes, model.pads, model.strides, activation))
    # end_time = time.time() - start_time
    # print('no numb time: ', end_time)
    
    # print(type(sample_results))
    
    # 3. 使用 numpy 的统计函数得到每个节点上的最值 or  中位数
    stratygy_map_max = []
    stratygy_map_min = []
    layer_num = len(sample_results[0])
    
    for nn_layer in range(layer_num):
        # if nn_layer < layer_num - 1:
        #     continue
        shape = (sample_num, sample_results[0][nn_layer].shape[0], sample_results[0][nn_layer].shape[1], sample_results[0][nn_layer].shape[2])
        t = np.zeros(shape, dtype=np.float32)
        # for i in range(sample_results[0][nn_layer].shape[2]):
        #     tmp_list = []
        #     for index in range(sample_num):
        #         # t[index] = sample_results[index][nn_layer]
        #         tmp_list.append(sample_results[index][nn_layer][0][0][i])
            
        #     plt.hist(tmp_list, color="g", histtype="bar", rwidth=10, alpha=0.4)
        #     plt.savefig('./test_2_'+str(i)+'.jpg')
        #     plt.clf()
        for index in range(sample_num):
            t[index] = sample_results[index][nn_layer]
        #median_ans = np.median(t, axis=0)

        max_ans = np.amax(t, axis=0)
        min_ans = np.amin(t, axis=0)
        #ans = median_ans.copy().astype(np.float32)
        max_ans = max_ans.copy().astype(np.float32)
        min_ans = min_ans.copy().astype(np.float32)
        #print(ans.dtype)
        #print(ans.shape)
        stratygy_map_max.append(max_ans)
        stratygy_map_min.append(min_ans)
    
    used_time = time.time() - start_time

    print("The upper bound of each neuron from Sampling:")
    print(stratygy_map_max)
    print("The lower bound of each neuron from Sampling:")
    print(stratygy_map_min)
    print("Time used for computing: %.2f" %(used_time))
    





    
    # activation = 'sigmoid'
    # eps = 0.005
    # under_LB = inputs[0] - eps
    # under_UB = inputs[0] + eps
    # under_LBs = [under_LB]
    # under_UBs = [under_UB]
    
    # under_LBs, under_UBs = obtain_under_range(under_LBs, under_UBs, model.weights, model.biases, model.shapes, model.pads, model.strides, inputs[0], activation, init=True)
    
    # for t in range(10):
    #     x0 = random.uniform(under_LBs[0], under_UBs[0])
    #     under_LBs, under_UBs = obtain_under_range(under_LBs, under_UBs, model.weights, model.biases, model.shapes, model.pads, model.strides, x0, activation)
    
    # print(under_LBs)
    # print(under_UBs)