from numba import njit, jit
import numpy as np
# import matplotlib.pyplot as plt
import os

from cnn_bounds_core import pool, conv, conv_bound, conv_full, conv_bound_full, pool_linear_bounds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, Lambda, ZeroPadding2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, InputLayer, BatchNormalization, Reshape, Subtract
from tensorflow.keras.models import load_model
import tensorflow.keras as keras

from train_myself_model import ResidualStart, ResidualStart2
import tensorflow as tf
from utils import generate_data_myself, generate_data
import time
import datetime
from activations import *
from underestimated_range import *
# from solve import *
linear_bounds = None

import random

def loss(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)
class Model:
    def __init__(self, model, inp_shape = (28,28,1), eran_cnn=False, vnn_comp=False):
        self.shapes = []
        self.sizes = []
        self.weights = []
        self.biases = []
        self.pads = []
        self.strides = []
        self.types = []
        self.model = model
        
        cur_shape = inp_shape
        print('cur_shape:', cur_shape)
        self.shapes.append(cur_shape)
        i = 0
        if eran_cnn:
            i = 5
        while i < len(model.layers):
            layer = model.layers[i]
            i += 1
            print(cur_shape)
            weights = layer.get_weights()
            if type(layer) == Conv2D:
                print('conv')
                if len(weights) == 1:
                    W = weights[0].astype(np.float32)
                    b = np.zeros(W.shape[-1], dtye=np.float32)
                else:
                    W, b = weights
                    W = W.astype(np.float32)
                    b = b.astype(np.float32)
                padding = layer.get_config()['padding']
                stride = layer.get_config()['strides']
                pad = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                if padding == 'same':
                    desired_h = int(np.ceil(cur_shape[0]/stride[0]))
                    desired_w = int(np.ceil(cur_shape[0]/stride[1]))
                    total_padding_h = stride[0]*(desired_h-1)+W.shape[0]-cur_shape[0]
                    total_padding_w = stride[1]*(desired_w-1)+W.shape[1]-cur_shape[1]
                    pad = (int(np.floor(total_padding_h/2)),int(np.ceil(total_padding_h/2)),int(np.floor(total_padding_w/2)),int(np.ceil(total_padding_w/2)))
                else:
                    if (i >= 2) and (type(model.layers[i-2])==ZeroPadding2D):
                        pad = (1, 1, 1, 1) # just adept models trained by eran
                cur_shape = (int((cur_shape[0]+pad[0]+pad[1]-W.shape[0])/stride[0])+1, int((cur_shape[1]+pad[2]+pad[3]-W.shape[1])/stride[1])+1, W.shape[-1])
                if vnn_comp:
                    W = W.transpose((0, 2, 3, 1))
                W = np.ascontiguousarray(W.transpose((3,0,1,2)).astype(np.float32))
                b = np.ascontiguousarray(b.astype(np.float32))
                self.types.append('conv')
                self.sizes.append(None)
                self.strides.append(stride)
                self.pads.append(pad)
                self.shapes.append(cur_shape)
                self.weights.append(W)
                self.biases.append(b)
            elif type(layer) == GlobalAveragePooling2D:
                print('global avg pool')
                b = np.zeros(cur_shape[-1], dtype=np.float32)
                W = np.zeros((cur_shape[0],cur_shape[1],cur_shape[2],cur_shape[2]), dtype=np.float32)
                for f in range(W.shape[2]):
                    W[:,:,f,f] = 1/(cur_shape[0]*cur_shape[1])
                pad = (0,0,0,0)
                stride = ((1,1))
                cur_shape = (1,1,cur_shape[2])
                W = np.ascontiguousarray(W.transpose((3,0,1,2)).astype(np.float32))
                b = np.ascontiguousarray(b.astype(np.float32))
                self.types.append('conv')
                self.sizes.append(None)
                self.strides.append(stride)
                self.pads.append(pad)
                self.shapes.append(cur_shape)
                self.weights.append(W)
                self.biases.append(b)
            elif type(layer) == AveragePooling2D:
                print('avg pool')
                b = np.zeros(cur_shape[-1], dtype=np.float32)
                padding = layer.get_config()['padding']
                pool_size = layer.get_config()['pool_size']
                stride = layer.get_config()['strides']
                W = np.zeros((pool_size[0],pool_size[1],cur_shape[2],cur_shape[2]), dtype=np.float32)
                for f in range(W.shape[2]):
                    W[:,:,f,f] = 1/(pool_size[0]*pool_size[1])
                pad = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                if padding == 'same':
                    desired_h = int(np.ceil(cur_shape[0]/stride[0]))
                    desired_w = int(np.ceil(cur_shape[0]/stride[1]))
                    total_padding_h = stride[0]*(desired_h-1)+pool_size[0]-cur_shape[0]
                    total_padding_w = stride[1]*(desired_w-1)+pool_size[1]-cur_shape[1]
                    pad = (int(np.floor(total_padding_h/2)),int(np.ceil(total_padding_h/2)),int(np.floor(total_padding_w/2)),int(np.ceil(total_padding_w/2)))
                cur_shape = (int((cur_shape[0]+pad[0]+pad[1]-pool_size[0])/stride[0])+1, int((cur_shape[1]+pad[2]+pad[3]-pool_size[1])/stride[1])+1, cur_shape[2])
                W = np.ascontiguousarray(W.transpose((3,0,1,2)).astype(np.float32))
                b = np.ascontiguousarray(b.astype(np.float32))
                self.types.append('conv')
                self.sizes.append(None)
                self.strides.append(stride)
                self.pads.append(pad)
                self.shapes.append(cur_shape)
                self.weights.append(W)
                self.biases.append(b)
            elif type(layer) == Activation or type(layer) == Lambda:
                if type(layer) == Lambda:
                    if eran_cnn:
                        continue
                print('activation')
                self.types.append('relu')
                self.sizes.append(None)
                self.strides.append(None)
                self.pads.append(None)
                self.shapes.append(cur_shape)
                self.weights.append(None)
                self.biases.append(None)
            elif type(layer) == InputLayer:
                print('input')
            elif type(layer) == BatchNormalization:
                print('batch normalization')
                gamma, beta, mean, std = weights
                std = np.sqrt(std+0.001) #Avoids zero division
                a = gamma/std
                b = -gamma*mean/std+beta
                self.weights[-1] = a*self.weights[-1]
                self.biases[-1] = a*self.biases[-1]+b
            elif type(layer) == Dense:
                print('FC')
                W, b = weights
                b = b.astype(np.float32)
                W = W.reshape(list(cur_shape)+[W.shape[-1]]).astype(np.float32)
                cur_shape = (1,1,W.shape[-1])
                W = np.ascontiguousarray(W.transpose((3,0,1,2)).astype(np.float32))
                b = np.ascontiguousarray(b.astype(np.float32))
                self.types.append('conv')
                self.sizes.append(None)
                self.strides.append((1,1))
                self.pads.append((0,0,0,0))
                self.shapes.append(cur_shape)
                self.weights.append(W)
                self.biases.append(b)
            elif type(layer) == Dropout:
                print('dropout')
            elif type(layer) == MaxPooling2D:
                print('pool')
                pool_size = layer.get_config()['pool_size']
                stride = layer.get_config()['strides']
                padding = layer.get_config()['padding']
                pad = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                if padding == 'same':
                    desired_h = int(np.ceil(cur_shape[0]/stride[0]))
                    desired_w = int(np.ceil(cur_shape[0]/stride[1]))
                    total_padding_h = stride[0]*(desired_h-1)+pool_size[0]-cur_shape[0]
                    total_padding_w = stride[1]*(desired_w-1)+pool_size[1]-cur_shape[1]
                    pad = (int(np.floor(total_padding_h/2)),int(np.ceil(total_padding_h/2)),int(np.floor(total_padding_w/2)),int(np.ceil(total_padding_w/2)))
                cur_shape = (int((cur_shape[0]+pad[0]+pad[1]-pool_size[0])/stride[0])+1, int((cur_shape[1]+pad[2]+pad[3]-pool_size[1])/stride[1])+1, cur_shape[2])
                self.types.append('pool')
                self.sizes.append(pool_size)
                self.strides.append(stride)
                self.pads.append(pad)
                self.shapes.append(cur_shape)
                self.weights.append(None)
                self.biases.append(None)
            elif type(layer) == Flatten:
                print('flatten')
            elif type(layer) == Reshape:
                print('reshape')
            elif type(layer) == ResidualStart2:
                print('basic block 2')
                conv1 = model.layers[i]
                bn1 = model.layers[i+1]
                conv2 = model.layers[i+3]
                conv3 = model.layers[i+4]
                bn2 = model.layers[i+5]
                bn3 = model.layers[i+6]
                i = i+8

                W1, bias1 = conv1.get_weights()
                W2, bias2 = conv2.get_weights()
                W3, bias3 = conv3.get_weights()
                
                gamma1, beta1, mean1, std1 = bn1.get_weights()
                std1 = np.sqrt(std1+0.001) #Avoids zero division
                a1 = gamma1/std1
                b1 = gamma1*mean1/std1+beta1
                W1 = a1*W1
                bias1 = a1*bias1+b1
                
                gamma2, beta2, mean2, std2 = bn2.get_weights()
                std2 = np.sqrt(std2+0.001) #Avoids zero division
                a2 = gamma2/std2
                b2 = gamma2*mean2/std2+beta2
                W2 = a2*W2
                bias2 = a2*bias2+b2
                 
                gamma3, beta3, mean3, std3 = bn3.get_weights()
                std3 = np.sqrt(std3+0.001) #Avoids zero division
                a3 = gamma3/std3
                b3 = gamma3*mean3/std3+beta3
                W3 = a3*W3
                bias3 = a3*bias3+b3

                padding1 = conv1.get_config()['padding']
                stride1 = conv1.get_config()['strides']
                pad1 = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                if padding1 == 'same':
                    desired_h = int(np.ceil(cur_shape[0]/stride1[0]))
                    desired_w = int(np.ceil(cur_shape[0]/stride1[1]))
                    total_padding_h = stride1[0]*(desired_h-1)+W1.shape[0]-cur_shape[0]
                    total_padding_w = stride1[1]*(desired_w-1)+W1.shape[1]-cur_shape[1]
                    pad1 = (int(np.floor(total_padding_h/2)),int(np.ceil(total_padding_h/2)),int(np.floor(total_padding_w/2)),int(np.ceil(total_padding_w/2)))
                cur_shape = (int((cur_shape[0]+pad1[0]+pad1[1]-W1.shape[0])/stride1[0])+1, int((cur_shape[1]+pad1[2]+pad1[3]-W1.shape[1])/stride1[1])+1, W1.shape[3])

                padding2 = conv2.get_config()['padding']
                stride2 = conv2.get_config()['strides']
                pad2 = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                if padding2 == 'same':
                    desired_h = int(np.ceil(cur_shape[0]/stride2[0]))
                    desired_w = int(np.ceil(cur_shape[0]/stride2[1]))
                    total_padding_h = stride2[0]*(desired_h-1)+W2.shape[0]-cur_shape[0]
                    total_padding_w = stride2[1]*(desired_w-1)+W2.shape[1]-cur_shape[1]
                    pad2 = (int(np.floor(total_padding_h/2)),int(np.ceil(total_padding_h/2)),int(np.floor(total_padding_w/2)),int(np.ceil(total_padding_w/2)))

                padding3 = conv3.get_config()['padding']
                stride3 = conv3.get_config()['strides']
                pad3 = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                if padding3 == 'same':
                    desired_h = int(np.ceil(cur_shape[0]/stride3[0]))
                    desired_w = int(np.ceil(cur_shape[0]/stride3[1]))
                    total_padding_h = stride3[0]*(desired_h-1)+W3.shape[0]-cur_shape[0]
                    total_padding_w = stride3[1]*(desired_w-1)+W3.shape[1]-cur_shape[1]
                    pad3 = (int(np.floor(total_padding_h/2)),int(np.ceil(total_padding_h/2)),int(np.floor(total_padding_w/2)),int(np.ceil(total_padding_w/2)))

                W1 = np.ascontiguousarray(W1.transpose((3,0,1,2)).astype(np.float32))
                bias1 = np.ascontiguousarray(bias1.astype(np.float32))
                W2 = np.ascontiguousarray(W2.transpose((3,0,1,2)).astype(np.float32))
                bias2 = np.ascontiguousarray(bias2.astype(np.float32))
                W3 = np.ascontiguousarray(W3.transpose((3,0,1,2)).astype(np.float32))
                bias3 = np.ascontiguousarray(bias3.astype(np.float32))
                self.types.append('basic_block_2')
                self.sizes.append(None)
                self.strides.append((stride1, stride2, stride3))
                self.pads.append((pad1, pad2, pad3))
                self.shapes.append(cur_shape)
                self.weights.append((W1, W2, W3))
                self.biases.append((bias1, bias2, bias3))
            elif type(layer) == ResidualStart:
                print('basic block')
                conv1 = model.layers[i]
                bn1 = model.layers[i+1]
                conv2 = model.layers[i+3]
                bn2 = model.layers[i+4]
                i = i+6

                W1, bias1 = conv1.get_weights()
                W2, bias2 = conv2.get_weights()
                
                gamma1, beta1, mean1, std1 = bn1.get_weights()
                std1 = np.sqrt(std1+0.001) #Avoids zero division
                a1 = gamma1/std1
                b1 = gamma1*mean1/std1+beta1
                W1 = a1*W1
                bias1 = a1*bias1+b1
                
                gamma2, beta2, mean2, std2 = bn2.get_weights()
                std2 = np.sqrt(std2+0.001) #Avoids zero division
                a2 = gamma2/std2
                b2 = gamma2*mean2/std2+beta2
                W2 = a2*W2
                bias2 = a2*bias2+b2

                padding1 = conv1.get_config()['padding']
                stride1 = conv1.get_config()['strides']
                pad1 = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                if padding1 == 'same':
                    desired_h = int(np.ceil(cur_shape[0]/stride1[0]))
                    desired_w = int(np.ceil(cur_shape[0]/stride1[1]))
                    total_padding_h = stride1[0]*(desired_h-1)+W1.shape[0]-cur_shape[0]
                    total_padding_w = stride1[1]*(desired_w-1)+W1.shape[1]-cur_shape[1]
                    pad1 = (int(np.floor(total_padding_h/2)),int(np.ceil(total_padding_h/2)),int(np.floor(total_padding_w/2)),int(np.ceil(total_padding_w/2)))
                cur_shape = (int((cur_shape[0]+pad1[0]+pad1[1]-W1.shape[0])/stride1[0])+1, int((cur_shape[1]+pad1[2]+pad1[3]-W1.shape[1])/stride1[1])+1, W1.shape[3])

                padding2 = conv2.get_config()['padding']
                stride2 = conv2.get_config()['strides']
                pad2 = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                if padding2 == 'same':
                    desired_h = int(np.ceil(cur_shape[0]/stride2[0]))
                    desired_w = int(np.ceil(cur_shape[0]/stride2[1]))
                    total_padding_h = stride2[0]*(desired_h-1)+W2.shape[0]-cur_shape[0]
                    total_padding_w = stride2[1]*(desired_w-1)+W2.shape[1]-cur_shape[1]
                    pad2 = (int(np.floor(total_padding_h/2)),int(np.ceil(total_padding_h/2)),int(np.floor(total_padding_w/2)),int(np.ceil(total_padding_w/2)))

                W1 = np.ascontiguousarray(W1.transpose((3,0,1,2)).astype(np.float32))
                bias1 = np.ascontiguousarray(bias1.astype(np.float32))
                W2 = np.ascontiguousarray(W2.transpose((3,0,1,2)).astype(np.float32))
                bias2 = np.ascontiguousarray(bias2.astype(np.float32))
                self.types.append('basic_block')
                self.sizes.append(None)
                self.strides.append((stride1, stride2))
                self.pads.append((pad1, pad2))
                self.shapes.append(cur_shape)
                self.weights.append((W1, W2))
                self.biases.append((bias1, bias2))
            elif type(layer) == Subtract:
                print('Subtract')
                pass
            elif type(layer) == ZeroPadding2D:
                print('ZeroPadding2D')
                pass
            else:
                print(str(type(layer)))
                raise ValueError('Invalid Layer Type')
        print(cur_shape)

    def predict(self, data):
        return self.model(data)


@njit
def UL_conv_bound(A, B, pad, stride, shape, W, b, inner_pad, inner_stride, inner_shape):
    A_new = np.zeros((A.shape[0], A.shape[1], A.shape[2], inner_stride[0]*(A.shape[3]-1)+W.shape[1], inner_stride[1]*(A.shape[4]-1)+W.shape[2], W.shape[3]), dtype=np.float32)
    B_new = B.copy()
    assert A.shape[5] == W.shape[0]   


    for x in range(A_new.shape[0]):
        p_start = np.maximum(0, pad[0]-stride[0]*x)
        p_end = np.minimum(A.shape[3], shape[0]+pad[0]-stride[0]*x)
        t_start = np.maximum(0, -stride[0]*inner_stride[0]*x+inner_stride[0]*pad[0]+inner_pad[0])
        t_end = np.minimum(A_new.shape[3], inner_shape[0]-stride[0]*inner_stride[0]*x+inner_stride[0]*pad[0]+inner_pad[0])
        for y in range(A_new.shape[1]):
            q_start = np.maximum(0, pad[2]-stride[1]*y)
            q_end = np.minimum(A.shape[4], shape[1]+pad[2]-stride[1]*y)
            u_start = np.maximum(0, -stride[1]*inner_stride[1]*y+inner_stride[1]*pad[2]+inner_pad[2])
            u_end = np.minimum(A_new.shape[4], inner_shape[1]-stride[1]*inner_stride[1]*y+inner_stride[1]*pad[2]+inner_pad[2])
            for t in range(t_start, t_end):
                for u in range(u_start, u_end):
                    for p in range(p_start, p_end):
                        for q in range(q_start, q_end):
                            if 0<=t-inner_stride[0]*p<W.shape[1] and 0<=u-inner_stride[1]*q<W.shape[2]:
                                A_new[x,y,:,t,u,:] += np.dot(A[x,y,:,p,q,:],W[:,t-inner_stride[0]*p,u-inner_stride[1]*q,:])
            for p in range(p_start, p_end):
                for q in range(q_start, q_end):
                    B_new[x,y,:] += np.dot(A[x,y,:,p,q,:],b)
    return A_new, B_new

basic_block_2_cache = {}
def UL_basic_block_2_bound(A, B, pad, stride, W1, W2, W3, b1, b2, b3, pad1, pad2, pad3, stride1, stride2, stride3, upper=True):
    LB, UB = basic_block_2_cache[np.sum(W1)]
    A1, B1 = UL_conv_bound(A, B, np.asarray(pad), np.asarray(stride), np.asarray(UB.shape), W2, b2, np.asarray(pad2), np.asarray(stride2), np.asarray(UB.shape))
    inter_pad = (stride2[0]*pad[0]+pad2[0], stride2[0]*pad[1]+pad2[1], stride2[1]*pad[2]+pad2[2], stride2[1]*pad[3]+pad2[3])
    inter_stride = (stride2[0]*stride[0], stride2[1]*stride[1])
    alpha_u, alpha_l, beta_u, beta_l = linear_bounds(LB, UB)
    if upper:
        A1, B1 = UL_relu_bound(A1, B1, np.asarray(inter_pad), np.asarray(inter_stride), alpha_u, alpha_l, beta_u, beta_l)
    else:
        A1, B1 = UL_relu_bound(A1, B1, np.asarray(inter_pad), np.asarray(inter_stride), alpha_l, alpha_u, beta_l, beta_u)
    A1, B1 = UL_conv_bound(A1, B1, np.asarray(inter_pad), np.asarray(inter_stride), np.asarray(UB.shape), W1, b1, np.asarray(pad1), np.asarray(stride1), np.asarray(UB.shape))
    A2, B2 = UL_conv_bound(A, B, np.asarray(pad), np.asarray(stride), np.asarray(UB.shape), W3, b3, np.asarray(pad3), np.asarray(stride3), np.asarray(UB.shape))
    height_diff = A1.shape[3]-A2.shape[3]
    width_diff = A1.shape[4]-A2.shape[4]
    assert height_diff % 2 == 0
    assert width_diff % 2 == 0
    d_h = height_diff//2
    d_w = width_diff//2
    A1[:,:,:,d_h:A1.shape[3]-d_h,d_w:A1.shape[4]-d_w,:] += A2
    return A1, B1+B2-B

basic_block_cache = {}
def UL_basic_block_bound(A, B, pad, stride, W1, W2, b1, b2, pad1, pad2, stride1, stride2, upper=True):
    LB, UB = basic_block_cache[np.sum(W1)]
    A1, B1 = UL_conv_bound(A, B, np.asarray(pad), np.asarray(stride), np.asarray(UB.shape), W2, b2, np.asarray(pad2), np.asarray(stride2), np.asarray(UB.shape))
    inter_pad = (stride2[0]*pad[0]+pad2[0], stride2[0]*pad[1]+pad2[1], stride2[1]*pad[2]+pad2[2], stride2[1]*pad[3]+pad2[3])
    inter_stride = (stride2[0]*stride[0], stride2[1]*stride[1])
    alpha_u, alpha_l, beta_u, beta_l = linear_bounds(LB, UB)
    if upper:
        A1, B1 = UL_relu_bound(A1, B1, np.asarray(inter_pad), np.asarray(inter_stride), alpha_u, alpha_l, beta_u, beta_l)
    else:
        A1, B1 = UL_relu_bound(A1, B1, np.asarray(inter_pad), np.asarray(inter_stride), alpha_l, alpha_u, beta_l, beta_u)
    A1, B1 = UL_conv_bound(A1, B1, np.asarray(inter_pad), np.asarray(inter_stride), np.asarray(UB.shape), W1, b1, np.asarray(pad1), np.asarray(stride1), np.asarray(UB.shape))
    height_diff = A1.shape[3]-A.shape[3]
    width_diff = A1.shape[4]-A.shape[4]
    assert height_diff % 2 == 0
    assert width_diff % 2 == 0
    d_h = height_diff//2
    d_w = width_diff//2
    A1[:,:,:,d_h:A1.shape[3]-d_h,d_w:A1.shape[4]-d_w,:] += A
    return A1, B1

@njit
def UL_relu_bound(A, B, pad, stride, alpha_u, alpha_l, beta_u, beta_l):
    A_new = np.zeros_like(A)
    A_plus = np.maximum(A, 0)
    A_minus = np.minimum(A, 0)
    B_new = B.copy()
    for x in range(A_new.shape[0]):
        p_start = np.maximum(0, pad[0]-stride[0]*x)
        p_end = np.minimum(A.shape[3], alpha_u.shape[0]+pad[0]-stride[0]*x)
        for y in range(A_new.shape[1]):
            q_start = np.maximum(0, pad[2]-stride[1]*y)
            q_end = np.minimum(A.shape[4], alpha_u.shape[1]+pad[2]-stride[1]*y)
            for z in range(A_new.shape[2]):
                for p in range(p_start, p_end):
                    for q in range(q_start, q_end):
                        for r in range(A.shape[5]):
                            A_new[x,y,z,p,q,r] += A_plus[x,y,z,p,q,r]*alpha_u[p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],r]
                            A_new[x,y,z,p,q,r] += A_minus[x,y,z,p,q,r]*alpha_l[p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],r]
                            B_new[x,y,z] += A_plus[x,y,z,p,q,r]*beta_u[p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],r]
                            B_new[x,y,z] += A_minus[x,y,z,p,q,r]*beta_l[p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],r]
    return A_new, B_new

@njit
def UL_pool_bound(A, B, pad, stride, pool_size, inner_pad, inner_stride, inner_shape, alpha_u, alpha_l, beta_u, beta_l):
    A_new = np.zeros((A.shape[0], A.shape[1], A.shape[2], inner_stride[0]*(A.shape[3]-1)+pool_size[0], inner_stride[1]*(A.shape[4]-1)+pool_size[1], A.shape[5]), dtype=np.float32)
    B_new = B.copy()
    A_plus = np.maximum(A, 0)
    A_minus = np.minimum(A, 0)

    for x in range(A_new.shape[0]):
        for y in range(A_new.shape[1]):
            for t in range(A_new.shape[3]):
                for u in range(A_new.shape[4]):
                    inner_index_x = t+stride[0]*inner_stride[0]*x-inner_stride[0]*pad[0]-inner_pad[0]
                    inner_index_y = u+stride[1]*inner_stride[1]*y-inner_stride[1]*pad[2]-inner_pad[2]
                    if 0<=inner_index_x<inner_shape[0] and 0<=inner_index_y<inner_shape[1]:
                        for p in range(A.shape[3]):
                            for q in range(A.shape[4]):
                                if 0<=t-inner_stride[0]*p<alpha_u.shape[0] and 0<=u-inner_stride[1]*q<alpha_u.shape[1] and 0<=p+stride[0]*x-pad[0]<alpha_u.shape[2] and 0<=q+stride[1]*y-pad[2]<alpha_u.shape[3]:
                                    A_new[x,y,:,t,u,:] += A_plus[x,y,:,p,q,:]*alpha_u[t-inner_stride[0]*p,u-inner_stride[1]*q,p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],:]
                                    A_new[x,y,:,t,u,:] += A_minus[x,y,:,p,q,:]*alpha_l[t-inner_stride[0]*p,u-inner_stride[1]*q,p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],:]
    B_new += conv_full(A_plus,beta_u,pad,stride) + conv_full(A_minus,beta_l,pad,stride)
    return A_new, B_new

def compute_bounds(weights, biases, out_shape, nlayer, x0, eps, p_n, pads, strides, sizes, types, LBs, UBs, activation, strategy_map_LB, strategy_map_UB, method='NeWise'):
    if types[nlayer-1] == 'relu':
        if activation == 'relu':
            return np.maximum(LBs[nlayer-1], 0), np.maximum(UBs[nlayer-1], 0), None, None, None, None, None, None
        elif activation == 'sigmoid':
            return sigmoid(LBs[nlayer-1]), sigmoid(UBs[nlayer-1]), None, None, None, None, None, None
        elif activation == 'tanh':
            return tanh(LBs[nlayer-1]), tanh(UBs[nlayer-1]), None, None, None, None, None, None
        elif activation == 'atan':
            return atan(LBs[nlayer-1]), atan(UBs[nlayer-1]), None, None, None, None, None, None
    elif types[nlayer-1] == 'conv':
        A_u = weights[nlayer-1].reshape((1, 1, weights[nlayer-1].shape[0], weights[nlayer-1].shape[1], weights[nlayer-1].shape[2], weights[nlayer-1].shape[3]))*np.ones((out_shape[0], out_shape[1], weights[nlayer-1].shape[0], weights[nlayer-1].shape[1], weights[nlayer-1].shape[2], weights[nlayer-1].shape[3]), dtype=np.float32)
        B_u = biases[nlayer-1]*np.ones((out_shape[0], out_shape[1], out_shape[2]), dtype=np.float32)
        A_l = A_u.copy()
        B_l = B_u.copy()
        pad = pads[nlayer-1]
        stride = strides[nlayer-1]
    elif types[nlayer-1] == 'pool':
        A_u = np.eye(out_shape[2]).astype(np.float32).reshape((1,1,out_shape[2],1,1,out_shape[2]))*np.ones((out_shape[0], out_shape[1], out_shape[2], 1,1,out_shape[2]), dtype=np.float32)
        B_u = np.zeros(out_shape, dtype=np.float32)
        A_l = A_u.copy()
        B_l = B_u.copy()
        pad = (0,0,0,0)
        stride = (1,1)
        alpha_u, alpha_l, beta_u, beta_l = pool_linear_bounds(LBs[nlayer-1], UBs[nlayer-1], np.asarray(pads[nlayer-1]), np.asarray(strides[nlayer-1]),  np.asarray(sizes[nlayer-1]))
        A_u, B_u = UL_pool_bound(A_u, B_u, np.asarray(pad), np.asarray(stride), np.asarray(sizes[nlayer-1]), np.asarray(pads[nlayer-1]), np.asarray(strides[nlayer-1]), np.asarray(LBs[nlayer-1].shape), alpha_u, alpha_l, beta_u, beta_l)
        A_l, B_l = UL_pool_bound(A_l, B_l, np.asarray(pad), np.asarray(stride), np.asarray(sizes[nlayer-1]), np.asarray(pads[nlayer-1]), np.asarray(strides[nlayer-1]), np.asarray(LBs[nlayer-1].shape), alpha_l, alpha_u, beta_l, beta_u)
        pad = pads[nlayer-1]
        stride = strides[nlayer-1]
    elif types[nlayer-1] == 'basic_block_2':
        W1, W2, W3 = weights[nlayer-1]
        b1, b2, b3 = biases[nlayer-1]
        pad1, pad2, pad3 = pads[nlayer-1]
        stride1, stride2, stride3 = strides[nlayer-1]
        LB, UB, A_u, A_l, B_u, B_l, pad, stride = compute_bounds(weights[:nlayer-1]+[W1], biases[:nlayer-1]+[b1], out_shape, nlayer, x0, eps, p_n, pads[:nlayer-1]+[pad1], strides[:nlayer-1]+[stride1], sizes, types[:nlayer-1]+['conv'], LBs, UBs, activation)
        basic_block_2_cache[np.sum(W1)] = (LB, UB)

        A_u = np.eye(out_shape[2]).astype(np.float32).reshape((1,1,out_shape[2],1,1,out_shape[2]))*np.ones((out_shape[0], out_shape[1], out_shape[2], 1,1,out_shape[2]), dtype=np.float32)
        B_u = np.zeros(out_shape, dtype=np.float32)
        A_l = A_u.copy()
        B_l = B_u.copy()
        pad = (0,0,0,0)
        stride = (1,1)
        A_u, B_u = UL_basic_block_2_bound(A_u, B_u, pad, stride, *weights[nlayer-1], *biases[nlayer-1], *pads[nlayer-1], *strides[nlayer-1], upper=True)
        A_l, B_l = UL_basic_block_2_bound(A_l, B_l, pad, stride, *weights[nlayer-1], *biases[nlayer-1], *pads[nlayer-1], *strides[nlayer-1], upper=False)
        inner_pad = pad3
        inner_stride = stride3
        pad = (inner_stride[0]*pad[0]+inner_pad[0], inner_stride[0]*pad[1]+inner_pad[1], inner_stride[1]*pad[2]+inner_pad[2], inner_stride[1]*pad[3]+inner_pad[3])
        stride = (inner_stride[0]*stride[0], inner_stride[1]*stride[1])
    elif types[nlayer-1] == 'basic_block':
        W1, W2 = weights[nlayer-1]
        b1, b2 = biases[nlayer-1]
        pad1, pad2 = pads[nlayer-1]
        stride1, stride2 = strides[nlayer-1]
        LB, UB , A_u, A_l, B_u, B_l, pad, stride = compute_bounds(weights[:nlayer-1]+[W1], biases[:nlayer-1]+[b1], out_shape, nlayer, x0, eps, p_n, pads[:nlayer-1]+[pad1], strides[:nlayer-1]+[stride1], sizes, types[:nlayer-1]+['conv'], LBs, UBs, activation)
        basic_block_cache[np.sum(W1)] = (LB, UB)

        A_u = np.eye(out_shape[2]).astype(np.float32).reshape((1,1,out_shape[2],1,1,out_shape[2]))*np.ones((out_shape[0], out_shape[1], out_shape[2], 1,1,out_shape[2]), dtype=np.float32)
        B_u = np.zeros(out_shape, dtype=np.float32)
        A_l = A_u.copy()
        B_l = B_u.copy()
        pad = (0,0,0,0)
        stride = (1,1)
        A_u, B_u = UL_basic_block_bound(A_u, B_u, pad, stride, *weights[nlayer-1], *biases[nlayer-1], *pads[nlayer-1], *strides[nlayer-1], upper=True)
        A_l, B_l = UL_basic_block_bound(A_l, B_l, pad, stride, *weights[nlayer-1], *biases[nlayer-1], *pads[nlayer-1], *strides[nlayer-1], upper=False)
    
    for i in range(nlayer-2, -1, -1):
        if types[i] == 'conv':
            A_u, B_u = UL_conv_bound(A_u, B_u, np.asarray(pad), np.asarray(stride), np.asarray(UBs[i+1].shape), weights[i], biases[i], np.asarray(pads[i]), np.asarray(strides[i]), np.asarray(UBs[i].shape))
            A_l, B_l = UL_conv_bound(A_l, B_l, np.asarray(pad), np.asarray(stride), np.asarray(LBs[i+1].shape), weights[i], biases[i], np.asarray(pads[i]), np.asarray(strides[i]), np.asarray(LBs[i].shape))
            pad = (strides[i][0]*pad[0]+pads[i][0], strides[i][0]*pad[1]+pads[i][1], strides[i][1]*pad[2]+pads[i][2], strides[i][1]*pad[3]+pads[i][3])
            stride = (strides[i][0]*stride[0], strides[i][1]*stride[1])
        elif types[i] == 'pool':
            alpha_u, alpha_l, beta_u, beta_l = pool_linear_bounds(LBs[i], UBs[i], np.asarray(pads[i]), np.asarray(strides[i]),  np.asarray(sizes[i]))
            A_u, B_u = UL_pool_bound(A_u, B_u, np.asarray(pad), np.asarray(stride), np.asarray(sizes[i]), np.asarray(pads[i]), np.asarray(strides[i]), np.asarray(UBs[i].shape), alpha_u, alpha_l, beta_u, beta_l)
            A_l, B_l = UL_pool_bound(A_l, B_l, np.asarray(pad), np.asarray(stride), np.asarray(sizes[i]), np.asarray(pads[i]), np.asarray(strides[i]), np.asarray(LBs[i].shape), alpha_l, alpha_u, beta_l, beta_u)
            pad = (strides[i][0]*pad[0]+pads[i][0], strides[i][0]*pad[1]+pads[i][1], strides[i][1]*pad[2]+pads[i][2], strides[i][1]*pad[3]+pads[i][3])
            stride = (strides[i][0]*stride[0], strides[i][1]*stride[1])
        elif types[i] == 'relu':

            alpha_u, alpha_l, beta_u, beta_l = linear_bounds(LBs[i], UBs[i], strategy_map_LB[i], strategy_map_UB[i], method)
            A_u, B_u = UL_relu_bound(A_u, B_u, np.asarray(pad), np.asarray(stride), alpha_u, alpha_l, beta_u, beta_l)
            A_l, B_l = UL_relu_bound(A_l, B_l, np.asarray(pad), np.asarray(stride), alpha_l, alpha_u, beta_l, beta_u)
        elif types[i] == 'basic_block_2':
            A_u, B_u = UL_basic_block_2_bound(A_u, B_u, pad, stride, *weights[i], *biases[i], *pads[i], *strides[i], upper=True)
            A_l, B_l = UL_basic_block_2_bound(A_l, B_l, pad, stride, *weights[i], *biases[i], *pads[i], *strides[i], upper=False)
            inner_pad = pads[i][0]
            inner_stride = strides[i][0]
            pad = (inner_stride[0]*pad[0]+inner_pad[0], inner_stride[0]*pad[1]+inner_pad[1], inner_stride[1]*pad[2]+inner_pad[2], inner_stride[1]*pad[3]+inner_pad[3])
            stride = (inner_stride[0]*stride[0], inner_stride[1]*stride[1])
            inner_pad = pads[i][1]
            inner_stride = strides[i][1]
            pad = (inner_stride[0]*pad[0]+inner_pad[0], inner_stride[0]*pad[1]+inner_pad[1], inner_stride[1]*pad[2]+inner_pad[2], inner_stride[1]*pad[3]+inner_pad[3])
            stride = (inner_stride[0]*stride[0], inner_stride[1]*stride[1])
        elif types[i] == 'basic_block':
            A_u, B_u = UL_basic_block_bound(A_u, B_u, pad, stride, *weights[i], *biases[i], *pads[i], *strides[i], upper=True)
            A_l, B_l = UL_basic_block_bound(A_l, B_l, pad, stride, *weights[i], *biases[i], *pads[i], *strides[i], upper=False)
            inner_pad = pads[i][0]
            inner_stride = strides[i][0]
            pad = (inner_stride[0]*pad[0]+inner_pad[0], inner_stride[0]*pad[1]+inner_pad[1], inner_stride[1]*pad[2]+inner_pad[2], inner_stride[1]*pad[3]+inner_pad[3])
            stride = (inner_stride[0]*stride[0], inner_stride[1]*stride[1])
            inner_pad = pads[i][1]
            inner_stride = strides[i][1]
            pad = (inner_stride[0]*pad[0]+inner_pad[0], inner_stride[0]*pad[1]+inner_pad[1], inner_stride[1]*pad[2]+inner_pad[2], inner_stride[1]*pad[3]+inner_pad[3])
            stride = (inner_stride[0]*stride[0], inner_stride[1]*stride[1])
    LUB, UUB = conv_bound_full(A_u, B_u, np.asarray(pad), np.asarray(stride), x0, eps, p_n)
    LLB, ULB = conv_bound_full(A_l, B_l, np.asarray(pad), np.asarray(stride), x0, eps, p_n)
    return LLB, UUB, A_u, A_l, B_u, B_l, pad, stride

def find_output_bounds(weights, biases, shapes, pads, strides, sizes, types, x0, eps, p_n, activation, strategy_map_LB, strategy_map_UB, method='NeWise'):
    LBs = [x0-eps]
    UBs = [x0+eps]
    
    for i in range(1,len(weights)+1):
        # print('Layer ' + str(i))
        LB, UB, A_u, A_l, B_u, B_l, pad, stride = compute_bounds(weights, biases, shapes[i], i, x0, eps, p_n, pads, strides, sizes, types, LBs, UBs, activation, strategy_map_LB, strategy_map_UB, method)
        #print(UB)
        UBs.append(UB)
        LBs.append(LB)
        
        # printlog('--------------------' + str(i) + ' layer -------------------')
        # for one in range(LB.shape[0]):
        #     for two in range(LB.shape[1]):
        #         for three in range(LB.shape[2]):
        #             printlog("{:.5f}, {:.5f}".format(LB[one,two,three], UB[one,two,three]))
        #             # printlog("{:.5f}".format(UB[one,two,three]-LB[one,two,three]))
                    
    return LBs[-1], UBs[-1], A_u, A_l, B_u, B_l, pad, stride

#Warms up numba functions
def warmup(model, x, eps_0, p_n, activation, func, strategy_map_LB, strategy_map_UB):
    print('Warming up...')
    weights = model.weights[:-1]
    biases = model.biases[:-1]
    shapes = model.shapes[:-1]
    W, b, s = model.weights[-1], model.biases[-1], model.shapes[-1]
    last_weight = np.ascontiguousarray((W[0,:,:,:]).reshape([1]+list(W.shape[1:])),dtype=np.float32)
    weights.append(last_weight)
    biases.append(np.asarray([b[0]]))
    shapes.append((1,1,1))
    func(weights, biases, shapes, model.pads, model.strides, model.sizes, model.types, x, eps_0, p_n, activation, strategy_map_LB, strategy_map_UB)

ts = time.time()
timestr = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')
#Prints to log file
def printlog(s):
    print(s)
    print(s, file=open("logs/cnn_bounds_full_with_LP_"+timestr+".txt", "a"))

def run_verified_robustness_ratio(file_name, n_samples, p_n, q_n, data_from_local=True, method='NeWise', sample_num=0,
                         cnn_cert_model=False, vnn_comp_model=False, 
                         eran_fnn=False, eran_cnn=False,
                         activation = 'sigmoid', mnist=False, cifar=False, fashion_mnist=False, gtsrb=False, step_tmp = 0.45, eps=0.005):
    np.random.seed(1215)
    random.seed(1215)
    if cnn_cert_model:
        # keras_model = load_model(file_name, custom_objects={'fn':loss, 'ResidualStart':ResidualStart, 'ResidualStart2':ResidualStart2, 'tf':tf})
        keras_model = load_model(file_name, custom_objects={'fn':loss, 'tf':tf})
    else:
        keras_model = load_model(file_name)
    
    if cifar:
        model = Model(keras_model, inp_shape = (32,32,3))
    elif gtsrb:
        print('gtsrb')
        model = Model(keras_model, inp_shape = (48,48,3))
    else:
        if vnn_comp_model:
            model = Model(keras_model, inp_shape = (1, 784, 1), vnn_comp=True)
        elif eran_cnn:
            model = Model(keras_model, eran_cnn = eran_cnn)
        else:
            model = Model(keras_model)
    print('--------abstracted model-----------')
    
    global linear_bounds
    if activation == 'sigmoid':
        linear_bounds = sigmoid_linear_bounds
    elif activation == 'tanh':
        linear_bounds = tanh_linear_bounds
    elif activation == 'atan':
        linear_bounds = atan_linear_bounds

    dataset = ''
    
    if cifar:
        dataset = 'cifar10'
        inputs, targets, true_labels, true_ids, img_info = generate_data('cifar10', samples=n_samples, data_from_local=data_from_local, targeted=True, random_and_least_likely = True, target_type = 0b0010, predictor=model.model.predict, start=0, cnn_cert_model=cnn_cert_model, eran_fnn=eran_fnn)
    elif gtsrb:
        dataset = 'gtsrb'
        inputs, targets, true_labels, true_ids = generate_data_myself('gtsrb', model.model, samples=n_samples, start=0)
    elif fashion_mnist:
        dataset = 'fashion_mnist'
        inputs, targets, true_labels, true_ids, img_info = generate_data('fashion_mnist', samples=n_samples, targeted=True, random_and_least_likely = True, target_type = 0b0010, predictor=model.model.predict, start=0, cnn_cert_model=cnn_cert_model)
    else:
        dataset = 'mnist'
        inputs, targets, true_labels, true_ids, img_info = generate_data('mnist', samples=n_samples, data_from_local=data_from_local, targeted=True, 
                                                                         random_and_least_likely = True, target_type = 0b0010, 
                                                                         predictor=keras_model.predict, start=0, 
                                                                         cnn_cert_model=cnn_cert_model, vnn_comp_model=vnn_comp_model, 
                                                                         eran_fnn=eran_fnn, eran_cnn=eran_cnn)

        
    #0b01111 <- all
    #0b0010 <- random
    #0b0001 <- top2 
    #0b0100 <- least
        
    print('----------generated data---------')

    printlog('===========================================')
    printlog("model name = {}".format(file_name))
    
    # printlog('====================' + method + '=======================')
   
    total_images = 0
    # new
    # steps = 15
    # eps_0 = 0.05
    # summation = 0
    # eps_total = np.zeros((len(inputs)), dtype=np.float32)
    
    strategy_map_LB = fnn_forward_propagation(inputs[0].astype(np.float32), model.weights, model.biases, model.shapes, model.pads, model.strides, 'sigmoid')
    strategy_map_UB = strategy_map_LB
    # new
    # warmup(model, inputs[0].astype(np.float32), eps_0, p_n, activation, find_output_bounds, strategy_map_LB, strategy_map_UB)
    warmup(model, inputs[0].astype(np.float32), eps, p_n, activation, find_output_bounds, strategy_map_LB, strategy_map_UB)
    
    sampling_total_time = 0
    verify_total_time = 0
    GD_total_time = 0
    
    start_time = time.time()
    robust = 0
    not_robust = 0
    unknown = 0
    for i in range(len(inputs)):
        # printlog('--- ' + method + ' relaxation: Computing eps for input image ' + str(i)+ '---')
        predict_label = np.argmax(true_labels[i])
        target_label = np.argmax(targets[i])
        
        #Perform binary search
        # new
        # log_eps = np.log(eps_0)
        # log_eps_min = -np.inf
        # log_eps_max = np.inf

        # for Gradient descent, compute sign of gradient of current input sample
        if method == 'gradient_descent':
            feed_x_0 = tf.convert_to_tensor(inputs[i:i+1].astype(np.float32), tf.float32)
            # print(tf.executing_eagerly())
            layer_num = len(keras_model.layers)
            GRAD_ALL = []

            GD_start_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(feed_x_0)
                x = feed_x_0
                flag = 0
                for cur_layer, layer in enumerate(keras_model.layers):
                    #print(cur_layer)
                    #print(type(keras_model.layers[cur_layer]))
                    GRAD = []
                    if type(layer) == Dense:
                        print("generating layer %d"%(cur_layer))
                        if flag == 1:
                            if activation == 'sigmoid':
                                x = tf.nn.sigmoid(x)
                            elif activation == 'tanh':
                                x = tf.nn.tanh(x)
                            elif activation == 'atan':
                                x = tf.atan(x)
                        elif flag == 0:
                            flag = 1
                        num_neurons = keras_model.layers[cur_layer].output_shape[-1]
                        weights = layer.get_weights()
                        W, b = weights
                        x = tf.reshape(x, [1, W.shape[0]])
                        x = tf.matmul(x, W) + b
                        for cur_neuron in range(num_neurons): 
                            grad_cur_neuron = tape.gradient(target=x[0][cur_neuron], sources=feed_x_0)
                            grad_cur_neuron = grad_cur_neuron.numpy()
                            GRAD.append(np.sign(grad_cur_neuron))
                    GRAD_ALL.append(GRAD)
            del tape
            
            GD_end_time = time.time() - GD_start_time
            print("sign of GD consumes: ", GD_end_time)
            GD_total_time += GD_end_time

        # new
        # for j in range(steps):
        if method == 'gradient_descent':
            GD_start_time = time.time()
            strategy_map_LB = []
            strategy_map_UB = []
            #strategy_map_mid = []
            cur_eps = eps

            strategy_map_LB.append(inputs[i].astype(np.float32) - cur_eps)
            strategy_map_UB.append(inputs[i].astype(np.float32) + cur_eps)
            #strategy_map_mid.append(inputs[i].astype(np.float32))
            feed_x_0 = inputs[i:i+1].astype(np.float32)

            for cur_layer in range(layer_num):
                #print(cur_layer)
                #print(type(keras_model.layers[cur_layer]))
                if type(keras_model.layers[cur_layer]) == Dense:
                    num_neurons = keras_model.layers[cur_layer].output_shape[-1]
                    shape = (1, 1, num_neurons)
                    ub_ans = np.zeros(shape, dtype=np.float32)
                    lb_ans = np.zeros(shape, dtype=np.float32)
                    #mid_ans = np.zeros(shape, dtype=np.float32)

                    #GD_1_step = GD_input(keras_model, cur_layer, activation)
                    #with tf.compat.v1.Session() as sess:
                    
                    for cur_neuron in range(num_neurons):
                        grad_cur_neuron = GRAD_ALL[cur_layer][cur_neuron]

                        x = copy.deepcopy(feed_x_0)
                        x -= cur_eps / 2.0 * grad_cur_neuron * step_tmp
                        x = np.clip(x, feed_x_0 - cur_eps, feed_x_0 + cur_eps) 
                        # x = np.clip(x, x_0_lower_bound, x_0_upper_bound)
                        #lb_ans[0][0][cur_neuron] = sess.run(GD_1_step.func, feed_dict = {GD_1_step.x_input: x})[0][cur_neuron]
                        lb_ans[0][0][cur_neuron] = fnn_forward_propagation(x[0], model.weights, model.biases, model.shapes, model.pads, model.strides, activation)[cur_layer][0][0][cur_neuron]

                        x = copy.deepcopy(feed_x_0)
                        x += cur_eps / 2.0 * grad_cur_neuron * step_tmp
                        x = np.clip(x, feed_x_0 - cur_eps, feed_x_0 + cur_eps) 
                        # x = np.clip(x, x_0_lower_bound, x_0_upper_bound)
                        #ub_ans[0][0][cur_neuron] = sess.run(GD_1_step.func, feed_dict = {GD_1_step.x_input: x})[0][cur_neuron]
                        ub_ans[0][0][cur_neuron] = fnn_forward_propagation(x[0], model.weights, model.biases, model.shapes, model.pads, model.strides, activation)[cur_layer][0][0][cur_neuron]
                    
                    ub_ans_copy = ub_ans.copy().astype(np.float32)
                    lb_ans_copy = lb_ans.copy().astype(np.float32)
                    #mid_ans_copy = mid_ans.copy().astype(np.float32)
                    strategy_map_UB.append(ub_ans_copy)
                    strategy_map_LB.append(lb_ans_copy)
                    #strategy_map_mid.append(mid_ans_copy)

                    if cur_layer != layer_num - 1:
                        if activation == 'sigmoid':
                            ub_ans_copy = sigmoid(ub_ans_copy)
                            lb_ans_copy = sigmoid(lb_ans_copy)
                        elif activation == 'tanh':
                            ub_ans_copy = tanh(ub_ans_copy)
                            lb_ans_copy = tanh(lb_ans_copy)
                        elif activation == 'atan':
                            ub_ans_copy = atan(ub_ans_copy)
                            lb_ans_copy = atan(lb_ans_copy)
                        ub_ans_copy = ub_ans_copy.copy().astype(np.float32)
                        lb_ans_copy = lb_ans_copy.copy().astype(np.float32)
                        strategy_map_UB.append(ub_ans_copy)
                        strategy_map_LB.append(lb_ans_copy)

            
            GD_end_time = time.time() - GD_start_time
            GD_total_time += GD_end_time
            # print('GD time: ', GD_total_time)

        if method == 'guided_by_endpoint':
            sampling_start_time = time.time()
            
            # 1. 产生 sample_num 个随机样本，使用 numpy.ndarray 结构存储，shape: (sample_num, input_shape)
            # sample_num = 300
            samples_from_ith_image_shape = (sample_num, inputs[i].shape[0], inputs[i].shape[1], inputs[i].shape[2])
            # lower_bound_for_sample = inputs[i] - np.exp(log_eps)
            # lower_bound_for_sample = np.clip(lower_bound_for_sample, x_0_lower_bound, x_0_upper_bound)
            # upper_bound_for_sample = inputs[i] + np.exp(log_eps)
            # upper_bound_for_sample = np.clip(upper_bound_for_sample, x_0_lower_bound, x_0_upper_bound)
            # samples_from_ith_image = np.random.uniform(lower_bound_for_sample, upper_bound_for_sample, samples_from_ith_image_shape).astype(np.float32)
            # new
            # samples_from_ith_image = np.random.uniform(inputs[i] - np.exp(log_eps), inputs[i] + np.exp(log_eps), samples_from_ith_image_shape).astype(np.float32)
            samples_from_ith_image = np.random.uniform(inputs[i] - eps, inputs[i] + eps, samples_from_ith_image_shape).astype(np.float32)
            
            # 2. 对这 sample_num 个随机样本进行前向传播，创建 sample_results 用于存储 sample_num 个采样点在每个节点的取值
            # 因为每一层的 shape 都不一样，所以对于每个采样点在每个节点的取值只能用 list 进行存储，
            # 进而 sample_num 个采样点的总结果也存储于 list 结构中，每个采样点在每一层的取值结果用 numpy.array 存储
            samples_from_ith_image_results = []
            
            for sample_index in range(sample_num):
                samples_from_ith_image_results.append(fnn_forward_propagation(samples_from_ith_image[sample_index], model.weights, model.biases, model.shapes, model.pads, model.strides, activation))
            
            # 3. 使用 numpy 的统计函数得到每个节点上的中位数
            strategy_map_LB = []
            strategy_map_UB = []
            layer_num = len(samples_from_ith_image_results[0])
            
            # print(layer_num)

            # printlog('====================== Sampling ======================')
                            
            for nn_layer in range(layer_num):
                t_shape = (sample_num, samples_from_ith_image_results[0][nn_layer].shape[0], samples_from_ith_image_results[0][nn_layer].shape[1], samples_from_ith_image_results[0][nn_layer].shape[2])
                t = np.zeros(t_shape, dtype=np.float32)
                for index in range(sample_num):
                    t[index] = samples_from_ith_image_results[index][nn_layer]
                lb_ans = np.amin(t, axis=0)
                lb_ans_copy = lb_ans.copy().astype(np.float32)
                strategy_map_LB.append(lb_ans_copy)
                ub_ans = np.amax(t, axis=0)
                ub_ans_copy = ub_ans.copy().astype(np.float32)
                strategy_map_UB.append(ub_ans_copy)
            
            
            sampling_end_time = time.time() - sampling_start_time
            sampling_total_time += sampling_end_time
            # print('sampling time: ', sampling_end_time)

        if method == 'guided_by_median':
            sampling_start_time = time.time()
            
            # 1. 产生 sample_num 个随机样本，使用 numpy.ndarray 结构存储，shape: (sample_num, input_shape)
            # sample_num = 300
            samples_from_ith_image_shape = (sample_num, inputs[i].shape[0], inputs[i].shape[1], inputs[i].shape[2])
            # lower_bound_for_sample = inputs[i] - np.exp(log_eps)
            # lower_bound_for_sample = np.clip(lower_bound_for_sample, x_0_lower_bound, x_0_upper_bound)
            # upper_bound_for_sample = inputs[i] + np.exp(log_eps)
            # upper_bound_for_sample = np.clip(upper_bound_for_sample, x_0_lower_bound, x_0_upper_bound)
            # samples_from_ith_image = np.random.uniform(lower_bound_for_sample, upper_bound_for_sample, samples_from_ith_image_shape).astype(np.float32)
            # new
            # samples_from_ith_image = np.random.uniform(inputs[i] - np.exp(log_eps), inputs[i] + np.exp(log_eps), samples_from_ith_image_shape).astype(np.float32)
            samples_from_ith_image = np.random.uniform(inputs[i] - eps, inputs[i] + eps, samples_from_ith_image_shape).astype(np.float32)
            
            # 2. 对这 sample_num 个随机样本进行前向传播，创建 sample_results 用于存储 sample_num 个采样点在每个节点的取值
            # 因为每一层的 shape 都不一样，所以对于每个采样点在每个节点的取值只能用 list 进行存储，
            # 进而 sample_num 个采样点的总结果也存储于 list 结构中，每个采样点在每一层的取值结果用 numpy.array 存储
            samples_from_ith_image_results = []
            
            for sample_index in range(sample_num):
                samples_from_ith_image_results.append(fnn_forward_propagation(samples_from_ith_image[sample_index], model.weights, model.biases, model.shapes, model.pads, model.strides, activation))
            
            # 3. 使用 numpy 的统计函数得到每个节点上的中位数
            strategy_map_LB = []
            strategy_map_UB = []
            layer_num = len(samples_from_ith_image_results[0])
            
            # print(layer_num)

            # printlog('====================== Sampling ======================')
                            
            for nn_layer in range(layer_num):
                t_shape = (sample_num, samples_from_ith_image_results[0][nn_layer].shape[0], samples_from_ith_image_results[0][nn_layer].shape[1], samples_from_ith_image_results[0][nn_layer].shape[2])
                t = np.zeros(t_shape, dtype=np.float32)
                for index in range(sample_num):
                    t[index] = samples_from_ith_image_results[index][nn_layer]
                lb_ans = np.median(t, axis=0)
                lb_ans_copy = lb_ans.copy().astype(np.float32)
                strategy_map_LB.append(lb_ans_copy)
            
            strategy_map_UB = strategy_map_LB
                
                # printlog('--------------------'+str(nn_layer)+' layer -------------------')
                # for one in range(lb_ans.shape[0]):
                #     for two in range(lb_ans.shape[1]):
                #         for three in range(lb_ans.shape[2]):
                #             printlog("{:.5f}, {:.5f}".format(lb_ans[one,two,three], ub_ans[one,two,three]))
                #             # printlog("{:.5f}".format(ub_ans[one,two,three]-lb_ans[one,two,three]))
            
            # printlog('====================== Sampling End ======================')
            #print(strategy_map_UB)
            sampling_end_time = time.time() - sampling_start_time
            sampling_total_time += sampling_end_time
        
        # printlog('====================== Overapproximation ======================')
        
        verify_start_time = time.time()
        if method == 'gradient_descent':
            method_core = 'guided_by_endpoint'
        else:
            method_core = method
        # LB_total, UB_total, _, _, _, _, _, _ = find_output_bounds(model.weights, model.biases, model.shapes, model.pads, model.strides, model.sizes, model.types, inputs[i].astype(np.float32), eps, p_n, activation, strategy_map_LB, strategy_map_UB, method_core)
        LB_total, UB_total, _, _, _, _, _, _ = find_output_bounds(model.weights, model.biases, model.shapes, model.pads, model.strides, model.sizes, model.types, inputs[i].astype(np.float32), eps, p_n, activation, strategy_map_LB, strategy_map_UB, method_core)

        is_robust = True
        find_conter_example = False
        for j in range(10):
            if j == predict_label:
                continue
            target_label = j
            print(f"Image {i}, predict_label = {predict_label}, target_label = {target_label}")
            
            if LB_total[0, 0, predict_label] - UB_total[0, 0, target_label] > 0:  # robust
                print(f"This target_label is robust, continue")
                continue
            else:
                is_robust = False
                break
            
            # # 使用gurobi求反例，弃用
            # # 未必鲁棒，所以找反例，除非找到一个反例，否则不break
            # weights = model.weights[:-1]
            # biases = model.biases[:-1]
            # shapes = model.shapes[:-1]
            # W, b, s = model.weights[-1], model.biases[-1], model.shapes[-1]
            # last_weight = (W[predict_label,:,:,:]-W[target_label,:,:,:]).reshape([1]+list(W.shape[1:]))
            # weights.append(last_weight)
            # biases.append(np.asarray([b[predict_label]-b[target_label]]))
            # shapes.append((1,1,1))
            # LB, UB, A_u, A_l, B_u, B_l, pad, stride = find_output_bounds(weights, biases, shapes, model.pads, model.strides, model.sizes, model.types, inputs[i].astype(np.float32), eps, p_n, activation, strategy_map_LB, strategy_map_UB, method_core)
            
            # # 如果所有lb都>0，则robust
            # # 如果其中一个lb<0，就寻找反例，找到了就是not robust
            # # 找不到就是unknown
            # distance_bt_pre_tar = LB[0][0][0]
            # if distance_bt_pre_tar > 0:
            #     print(f"This target_label is robust, continue")
            #     continue
            # else:
            #     is_robust = False
            #     # 寻找反例
            #     lp_model = new_model()
            #     lp_model, x = creat_var(lp_model, inputs[i], eps)
            #     shape = inputs[i].shape
            #     adv_image, min_val = get_solution_value(lp_model, x, shape, A_u, A_l, B_u, B_l, pad, stride, p_n, eps)
            #     if min_val <= 0:
            #         # label of potential conter-example
            #         a = adv_image[np.newaxis, :, :, :]
            #         aa = a.astype(np.float32)
            #         adv_label = np.argmax(np.squeeze(keras_model.predict(aa)))
            #         if adv_label != predict_label:
            #             not_robust += 1
            #             print(f"Image {i} is not robust, not robust number = {not_robust}")
            #             find_conter_example = True
            #             break
            #         else:
            #             print(f"This target_label is unknown")
            #     else:
            #         print(f"This target_label is unknown")
        if is_robust:
            robust += 1
            print(f"Image {i} is robust, robust number = {robust}")
        # elif not find_conter_example:
        #     unknown += 1
        #     print(f"Image {i} is unknown, unknown number = {unknown}")
        verify_total_time += time.time() - verify_start_time

    aver_GD_time = GD_total_time / len(inputs)
    aver_sample_time = sampling_total_time / len(inputs)
    aver_verify_time = verify_total_time / len(inputs)
    print("### Summary: eps={:.5f}, wrong_classification={}, robust_images={}, robust_rate={:.5f}, unknown={}, not_robust={}".format(eps, n_samples - len(inputs), robust, float(robust)/float(len(inputs)), unknown, not_robust))
    print(f"Time: aver_GD_time = {aver_GD_time}, aver_sample_time = {aver_sample_time}, aver_verify_time = {aver_verify_time}")




def run_certified_bounds(file_name, n_samples, p_n, q_n, data_from_local=True, method='NeWise', sample_num=0,
                         cnn_cert_model=False, vnn_comp_model=False, 
                         eran_fnn=False, eran_cnn=False,
                         activation = 'sigmoid', mnist=False, cifar=False, fashion_mnist=False, gtsrb=False, step_tmp = 0.45):
    np.random.seed(1215)
    random.seed(1215)
    if cnn_cert_model:
        # keras_model = load_model(file_name, custom_objects={'fn':loss, 'ResidualStart':ResidualStart, 'ResidualStart2':ResidualStart2, 'tf':tf})
        keras_model = load_model(file_name, custom_objects={'fn':loss, 'tf':tf})
    else:
        keras_model = load_model(file_name, custom_objects={'fn':loss, 'tf':tf})
    
    if cifar:
        model = Model(keras_model, inp_shape = (32,32,3))
    elif gtsrb:
        print('gtsrb')
        model = Model(keras_model, inp_shape = (48,48,3))
    else:
        if vnn_comp_model:
            model = Model(keras_model, inp_shape = (1, 784, 1), vnn_comp=True)
        elif eran_cnn:
            model = Model(keras_model, eran_cnn = eran_cnn)
        else:
            model = Model(keras_model)
    print('--------abstracted model-----------')
    
    global linear_bounds
    if activation == 'sigmoid':
        linear_bounds = sigmoid_linear_bounds
    elif activation == 'tanh':
        linear_bounds = tanh_linear_bounds
    elif activation == 'atan':
        linear_bounds = atan_linear_bounds

    dataset = ''
    
    if cifar:
        dataset = 'cifar10'
        inputs, targets, true_labels, true_ids, img_info = generate_data('cifar10', samples=n_samples, data_from_local=data_from_local, targeted=True, random_and_least_likely = True, target_type = 0b0010, predictor=model.model.predict, start=0, cnn_cert_model=cnn_cert_model, eran_fnn=eran_fnn)
    elif gtsrb:
        dataset = 'gtsrb'
        inputs, targets, true_labels, true_ids = generate_data_myself('gtsrb', model.model, samples=n_samples, start=0)
    elif fashion_mnist:
        dataset = 'fashion_mnist'
        inputs, targets, true_labels, true_ids, img_info = generate_data('fashion_mnist', samples=n_samples, targeted=True, random_and_least_likely = True, target_type = 0b0010, predictor=model.model.predict, start=0, cnn_cert_model=cnn_cert_model)
    else:
        dataset = 'mnist'
        inputs, targets, true_labels, true_ids, img_info = generate_data('mnist', samples=n_samples, data_from_local=data_from_local, targeted=True, 
                                                                         random_and_least_likely = True, target_type = 0b0010, 
                                                                         predictor=keras_model.predict, start=0, 
                                                                         cnn_cert_model=cnn_cert_model, vnn_comp_model=vnn_comp_model, 
                                                                         eran_fnn=eran_fnn, eran_cnn=eran_cnn)

    #0b01111 <- all
    #0b0010 <- random
    #0b0001 <- top2 
    #0b0100 <- least
        
    print('----------generated data---------')

    printlog('===========================================')
    printlog("model name = {}".format(file_name))
    
    # printlog('====================' + method + '=======================')
   
    total_images = 0
    steps = 15
    eps_0 = 0.05
    summation = 0
    eps_total = np.zeros((len(inputs)), dtype=np.float32)
    
    strategy_map_LB = fnn_forward_propagation(inputs[0].astype(np.float32), model.weights, model.biases, model.shapes, model.pads, model.strides, 'sigmoid')
    strategy_map_UB = strategy_map_LB
    warmup(model, inputs[0].astype(np.float32), eps_0, p_n, activation, find_output_bounds, strategy_map_LB, strategy_map_UB)
    
    sampling_total_time = 0
    verify_total_time = 0
    GD_total_time = 0
    
    start_time = time.time()
    for i in range(len(inputs)):
        # printlog('--- ' + method + ' relaxation: Computing eps for input image ' + str(i)+ '---')
        predict_label = np.argmax(true_labels[i])
        target_label = np.argmax(targets[i])
        
        #Perform binary search
        log_eps = np.log(eps_0)
        log_eps_min = -np.inf
        log_eps_max = np.inf

        # for Gradient descent, compute sign of gradient of current input sample
        if method == 'gradient_descent':
            feed_x_0 = tf.convert_to_tensor(inputs[i:i+1].astype(np.float32), tf.float32)
            # print(tf.executing_eagerly())
            layer_num = len(keras_model.layers)
            GRAD_ALL = []

            GD_start_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(feed_x_0)
                x = feed_x_0
                flag = 0
                for cur_layer, layer in enumerate(keras_model.layers):
                    #print(cur_layer)
                    #print(type(keras_model.layers[cur_layer]))
                    GRAD = []
                    if type(layer) == Dense:
                        print("generating layer %d"%(cur_layer))
                        if flag == 1:
                            if activation == 'sigmoid':
                                x = tf.nn.sigmoid(x)
                            elif activation == 'tanh':
                                x = tf.nn.tanh(x)
                            elif activation == 'atan':
                                x = tf.atan(x)
                        elif flag == 0:
                            flag = 1
                        num_neurons = keras_model.layers[cur_layer].output_shape[-1]
                        weights = layer.get_weights()
                        W, b = weights
                        x = tf.reshape(x, [1, W.shape[0]])
                        x = tf.matmul(x, W) + b
                        for cur_neuron in range(num_neurons): 
                            grad_cur_neuron = tape.gradient(target=x[0][cur_neuron], sources=feed_x_0)
                            grad_cur_neuron = grad_cur_neuron.numpy()
                            GRAD.append(np.sign(grad_cur_neuron))
                    GRAD_ALL.append(GRAD)
            del tape
            
            GD_end_time = time.time() - GD_start_time
            print("sign of GD consumes: ", GD_end_time)
            GD_total_time += GD_end_time

        
        for j in range(steps):
            if method == 'gradient_descent':
                GD_start_time = time.time()
                strategy_map_LB = []
                strategy_map_UB = []
                #strategy_map_mid = []
                cur_eps = np.exp(log_eps)

                strategy_map_LB.append(inputs[i].astype(np.float32) - cur_eps)
                strategy_map_UB.append(inputs[i].astype(np.float32) + cur_eps)
                #strategy_map_mid.append(inputs[i].astype(np.float32))
                feed_x_0 = inputs[i:i+1].astype(np.float32)

                for cur_layer in range(layer_num):
                    #print(cur_layer)
                    #print(type(keras_model.layers[cur_layer]))
                    if type(keras_model.layers[cur_layer]) == Dense:
                        num_neurons = keras_model.layers[cur_layer].output_shape[-1]
                        shape = (1, 1, num_neurons)
                        ub_ans = np.zeros(shape, dtype=np.float32)
                        lb_ans = np.zeros(shape, dtype=np.float32)
                        #mid_ans = np.zeros(shape, dtype=np.float32)

                        #GD_1_step = GD_input(keras_model, cur_layer, activation)
                        #with tf.compat.v1.Session() as sess:
                        
                        for cur_neuron in range(num_neurons):
                            grad_cur_neuron = GRAD_ALL[cur_layer][cur_neuron]

                            x = copy.deepcopy(feed_x_0)
                            x -= cur_eps / 2.0 * grad_cur_neuron * step_tmp
                            x = np.clip(x, feed_x_0 - cur_eps, feed_x_0 + cur_eps) 
                            # x = np.clip(x, x_0_lower_bound, x_0_upper_bound)
                            #lb_ans[0][0][cur_neuron] = sess.run(GD_1_step.func, feed_dict = {GD_1_step.x_input: x})[0][cur_neuron]
                            lb_ans[0][0][cur_neuron] = fnn_forward_propagation(x[0], model.weights, model.biases, model.shapes, model.pads, model.strides, activation)[cur_layer][0][0][cur_neuron]

                            x = copy.deepcopy(feed_x_0)
                            x += cur_eps / 2.0 * grad_cur_neuron * step_tmp
                            x = np.clip(x, feed_x_0 - cur_eps, feed_x_0 + cur_eps) 
                            # x = np.clip(x, x_0_lower_bound, x_0_upper_bound)
                            #ub_ans[0][0][cur_neuron] = sess.run(GD_1_step.func, feed_dict = {GD_1_step.x_input: x})[0][cur_neuron]
                            ub_ans[0][0][cur_neuron] = fnn_forward_propagation(x[0], model.weights, model.biases, model.shapes, model.pads, model.strides, activation)[cur_layer][0][0][cur_neuron]
                        
                        ub_ans_copy = ub_ans.copy().astype(np.float32)
                        lb_ans_copy = lb_ans.copy().astype(np.float32)
                        #mid_ans_copy = mid_ans.copy().astype(np.float32)
                        strategy_map_UB.append(ub_ans_copy)
                        strategy_map_LB.append(lb_ans_copy)
                        #strategy_map_mid.append(mid_ans_copy)

                        if cur_layer != layer_num - 1:
                            if activation == 'sigmoid':
                                ub_ans_copy = sigmoid(ub_ans_copy)
                                lb_ans_copy = sigmoid(lb_ans_copy)
                            elif activation == 'tanh':
                                ub_ans_copy = tanh(ub_ans_copy)
                                lb_ans_copy = tanh(lb_ans_copy)
                            elif activation == 'atan':
                                ub_ans_copy = atan(ub_ans_copy)
                                lb_ans_copy = atan(lb_ans_copy)
                            ub_ans_copy = ub_ans_copy.copy().astype(np.float32)
                            lb_ans_copy = lb_ans_copy.copy().astype(np.float32)
                            strategy_map_UB.append(ub_ans_copy)
                            strategy_map_LB.append(lb_ans_copy)

                
                GD_end_time = time.time() - GD_start_time
                GD_total_time += GD_end_time
                # print('GD time: ', GD_total_time)

            if method == 'guided_by_endpoint':
                sampling_start_time = time.time()
                
                # 1. 产生 sample_num 个随机样本，使用 numpy.ndarray 结构存储，shape: (sample_num, input_shape)
                # sample_num = 300
                samples_from_ith_image_shape = (sample_num, inputs[i].shape[0], inputs[i].shape[1], inputs[i].shape[2])
                # lower_bound_for_sample = inputs[i] - np.exp(log_eps)
                # lower_bound_for_sample = np.clip(lower_bound_for_sample, x_0_lower_bound, x_0_upper_bound)
                # upper_bound_for_sample = inputs[i] + np.exp(log_eps)
                # upper_bound_for_sample = np.clip(upper_bound_for_sample, x_0_lower_bound, x_0_upper_bound)
                # samples_from_ith_image = np.random.uniform(lower_bound_for_sample, upper_bound_for_sample, samples_from_ith_image_shape).astype(np.float32)
                samples_from_ith_image = np.random.uniform(inputs[i] - np.exp(log_eps), inputs[i] + np.exp(log_eps), samples_from_ith_image_shape).astype(np.float32)
                
                # 2. 对这 sample_num 个随机样本进行前向传播，创建 sample_results 用于存储 sample_num 个采样点在每个节点的取值
                # 因为每一层的 shape 都不一样，所以对于每个采样点在每个节点的取值只能用 list 进行存储，
                # 进而 sample_num 个采样点的总结果也存储于 list 结构中，每个采样点在每一层的取值结果用 numpy.array 存储
                samples_from_ith_image_results = []
                
                for sample_index in range(sample_num):
                    samples_from_ith_image_results.append(fnn_forward_propagation(samples_from_ith_image[sample_index], model.weights, model.biases, model.shapes, model.pads, model.strides, activation))
                
                # 3. 使用 numpy 的统计函数得到每个节点上的中位数
                strategy_map_LB = []
                strategy_map_UB = []
                layer_num = len(samples_from_ith_image_results[0])
                
                # print(layer_num)

                # printlog('====================== Sampling ======================')
                                
                for nn_layer in range(layer_num):
                    t_shape = (sample_num, samples_from_ith_image_results[0][nn_layer].shape[0], samples_from_ith_image_results[0][nn_layer].shape[1], samples_from_ith_image_results[0][nn_layer].shape[2])
                    t = np.zeros(t_shape, dtype=np.float32)
                    for index in range(sample_num):
                        t[index] = samples_from_ith_image_results[index][nn_layer]
                    lb_ans = np.amin(t, axis=0)
                    lb_ans_copy = lb_ans.copy().astype(np.float32)
                    strategy_map_LB.append(lb_ans_copy)
                    ub_ans = np.amax(t, axis=0)
                    ub_ans_copy = ub_ans.copy().astype(np.float32)
                    strategy_map_UB.append(ub_ans_copy)
                
                
                sampling_end_time = time.time() - sampling_start_time
                sampling_total_time += sampling_end_time
                # print('sampling time: ', sampling_end_time)

            if method == 'guided_by_median':
                sampling_start_time = time.time()
                
                # 1. 产生 sample_num 个随机样本，使用 numpy.ndarray 结构存储，shape: (sample_num, input_shape)
                # sample_num = 300
                samples_from_ith_image_shape = (sample_num, inputs[i].shape[0], inputs[i].shape[1], inputs[i].shape[2])
                # lower_bound_for_sample = inputs[i] - np.exp(log_eps)
                # lower_bound_for_sample = np.clip(lower_bound_for_sample, x_0_lower_bound, x_0_upper_bound)
                # upper_bound_for_sample = inputs[i] + np.exp(log_eps)
                # upper_bound_for_sample = np.clip(upper_bound_for_sample, x_0_lower_bound, x_0_upper_bound)
                # samples_from_ith_image = np.random.uniform(lower_bound_for_sample, upper_bound_for_sample, samples_from_ith_image_shape).astype(np.float32)
                samples_from_ith_image = np.random.uniform(inputs[i] - np.exp(log_eps), inputs[i] + np.exp(log_eps), samples_from_ith_image_shape).astype(np.float32)
                
                # 2. 对这 sample_num 个随机样本进行前向传播，创建 sample_results 用于存储 sample_num 个采样点在每个节点的取值
                # 因为每一层的 shape 都不一样，所以对于每个采样点在每个节点的取值只能用 list 进行存储，
                # 进而 sample_num 个采样点的总结果也存储于 list 结构中，每个采样点在每一层的取值结果用 numpy.array 存储
                samples_from_ith_image_results = []
                
                for sample_index in range(sample_num):
                    samples_from_ith_image_results.append(fnn_forward_propagation(samples_from_ith_image[sample_index], model.weights, model.biases, model.shapes, model.pads, model.strides, activation))
                
                # 3. 使用 numpy 的统计函数得到每个节点上的中位数
                strategy_map_LB = []
                strategy_map_UB = []
                layer_num = len(samples_from_ith_image_results[0])
                
                # print(layer_num)

                # printlog('====================== Sampling ======================')
                                
                for nn_layer in range(layer_num):
                    t_shape = (sample_num, samples_from_ith_image_results[0][nn_layer].shape[0], samples_from_ith_image_results[0][nn_layer].shape[1], samples_from_ith_image_results[0][nn_layer].shape[2])
                    t = np.zeros(t_shape, dtype=np.float32)
                    for index in range(sample_num):
                        t[index] = samples_from_ith_image_results[index][nn_layer]
                    lb_ans = np.median(t, axis=0)
                    lb_ans_copy = lb_ans.copy().astype(np.float32)
                    strategy_map_LB.append(lb_ans_copy)
                
                strategy_map_UB = strategy_map_LB
                    
                    # printlog('--------------------'+str(nn_layer)+' layer -------------------')
                    # for one in range(lb_ans.shape[0]):
                    #     for two in range(lb_ans.shape[1]):
                    #         for three in range(lb_ans.shape[2]):
                    #             printlog("{:.5f}, {:.5f}".format(lb_ans[one,two,three], ub_ans[one,two,three]))
                    #             # printlog("{:.5f}".format(ub_ans[one,two,three]-lb_ans[one,two,three]))
                
                # printlog('====================== Sampling End ======================')
                #print(strategy_map_UB)
                sampling_end_time = time.time() - sampling_start_time
                sampling_total_time += sampling_end_time
            
            # printlog('====================== Overapproximation ======================')
            
            verify_start_time = time.time()
            if method == 'gradient_descent':
                method_core = 'guided_by_endpoint'
            else:
                method_core = method
            LB_total, UB_total, _, _, _, _, _, _ = find_output_bounds(model.weights, model.biases, model.shapes, model.pads, model.strides, model.sizes, model.types, inputs[i].astype(np.float32), np.exp(log_eps), p_n, activation, strategy_map_LB, strategy_map_UB, method_core)
            distance_bt_pre_tar = LB_total[0][0][predict_label] - UB_total[0][0][target_label]
            print("Step {}, eps = {:.5f}, f_c_min - f_t_max = {:.6s}".format(j,np.exp(log_eps),str(distance_bt_pre_tar)))
            
            if distance_bt_pre_tar > 0: #Increase eps
                log_eps_min = log_eps
                log_eps = np.minimum(log_eps+1, (log_eps_max+log_eps_min)/2)
            else: #Decrease eps
                log_eps_max = log_eps
                log_eps = np.maximum(log_eps-1, (log_eps_max+log_eps_min)/2)
            verify_end_time = time.time() - verify_start_time
            verify_total_time += verify_end_time
            # print('verify time: ', verify_end_time)
            
        print("method = {}-{}, model = {}, image no = {}, true_id = {}, target_label = {}, true_label = {}, robustness = {:.5f}".format(method, activation,file_name, i, true_ids[i],target_label,predict_label,np.exp(log_eps_min)))
        summation += np.exp(log_eps_min)
        eps_total[i] = np.exp(log_eps_min)
    
    eps_avg = summation/len(inputs)
    aver_GD_time = GD_total_time / len(inputs)
    aver_sample_time = sampling_total_time / len(inputs)
    aver_verify_time = verify_total_time / len(inputs)
    printlog(f"sampling total time: {sampling_total_time}, GD total time: {GD_total_time}, len_inputs: {len(inputs)}")
    # printlog("[L0] method = {}-{}, model = {}, total images = {}, avg robustness = {:.5f}, avg runtime = {:.2f}".format(method, activation,file_name,len(inputs),eps_avg,aver_time))
    printlog("[L0] method = {}-{}, total images = {}, avg robustness = {:.5f}, avg verify runtime = {:.2f}, avg sample runtime = {:.2f}, avg GD runtime = {:.2f}, sample num = {}, step = {}".format(method, activation, len(inputs),eps_avg,aver_verify_time, aver_sample_time, aver_GD_time, sample_num, steps))
    printlog("[L0] method = {}-{}, robustness: mean = {:.5f}, std = {:.5f}, var = {:.5f}, max = {:.5f}, min = {:.5f}".format(method, activation, np.mean(eps_total), np.std(eps_total), np.var(eps_total), np.amax(eps_total), np.amin(eps_total)))




