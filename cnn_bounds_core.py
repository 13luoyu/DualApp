from numba import njit
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, InputLayer, BatchNormalization, Reshape
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist, cifar10
import tensorflow as tf


from utils import generate_data_myself, generate_data
import time
import datetime
from activations import *
from underestimated_range import *
import copy
# from solve import *
linear_bounds = None

import random

def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)
class CNNModel:
    def __init__(self, model, inp_shape = (28,28,1)):
        print('-----------', inp_shape, '---------')
        temp_weights = [layer.get_weights() for layer in model.layers]

        self.weights = []
        self.biases = []
        self.shapes = []
        self.pads = []
        self.strides = []
        self.model = model
        
        cur_shape = inp_shape
        self.shapes.append(cur_shape)
        for layer in model.layers:
            print(cur_shape)
            weights = layer.get_weights()
            if type(layer) == Conv2D:
                print('conv')
                if len(weights) == 1:
                    W = weights[0].astype(np.float32)
                    b = np.zeros(W.shape[-1], dtype=np.float32)
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
                cur_shape = (int((cur_shape[0]+pad[0]+pad[1]-W.shape[0])/stride[0])+1, int((cur_shape[1]+pad[2]+pad[3]-W.shape[1])/stride[1])+1, W.shape[-1])
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
                self.strides.append(stride)
                self.pads.append(pad)
                self.shapes.append(cur_shape)
                self.weights.append(W)
                self.biases.append(b)
            elif type(layer) == AveragePooling2D:
                print('avg pool')
                b = np.zeros(cur_shape[-1], dtype=np.float32)
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
                self.strides.append(stride)
                self.pads.append(pad)
                self.shapes.append(cur_shape)
                self.weights.append(W)
                self.biases.append(b)
            elif type(layer) == Activation:
                print('activation')
            elif type(layer) == Lambda:
                print('lambda')
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
                pad = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                if padding == 'same':
                    desired_h = int(np.ceil(cur_shape[0]/stride[0]))
                    desired_w = int(np.ceil(cur_shape[0]/stride[1]))
                    total_padding_h = stride[0]*(desired_h-1)+pool_size[0]-cur_shape[0]
                    total_padding_w = stride[1]*(desired_w-1)+pool_size[1]-cur_shape[1]
                    pad = (int(np.floor(total_padding_h/2)),int(np.ceil(total_padding_h/2)),int(np.floor(total_padding_w/2)),int(np.ceil(total_padding_w/2)))
                cur_shape = (int((cur_shape[0]+pad[0]+pad[1]-pool_size[0])/stride[0])+1, int((cur_shape[1]+pad[2]+pad[3]-pool_size[1])/stride[1])+1, cur_shape[2])
                self.strides.append(stride)
                self.pads.append(pad)
                self.shapes.append(cur_shape)
                self.weights.append(np.full(pool_size+(1,1),np.nan,dtype=np.float32))
                self.biases.append(np.full(1,np.nan,dtype=np.float32))
            elif type(layer) == Flatten:
                print('flatten')
            elif type(layer) == Reshape:
                print('reshape')
            else:
                print(str(type(layer)))
                raise ValueError('Invalid Layer Type')
        print(cur_shape)

        for i in range(len(self.weights)):
            self.weights[i] = np.ascontiguousarray(self.weights[i].transpose((3,0,1,2)).astype(np.float32))
            self.biases[i] = np.ascontiguousarray(self.biases[i].astype(np.float32))
    def predict(self, data):
        return self.model(data)


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
def pool(pool_size, x0, pad, stride):
    p_hl, p_hr, p_wl, p_wr = pad
    s_h, s_w = stride
    y0 = np.zeros((int((x0.shape[0]+p_hl+p_hr-pool_size[0])/s_h)+1, int((x0.shape[1]+p_wl+p_wr-pool_size[1])/s_w)+1, x0.shape[2]), dtype=np.float32)
    for x in range(y0.shape[0]):
        for y in range(y0.shape[1]):
            for r in range(y0.shape[2]):
                cropped = LB[s_h*x-p_hl:pool_size[0]+s_h*x-p_hl, s_w*y-p_wl:pool_size[1]+s_w*y-p_wl,r]
                y0[x,y,r] = cropped.max()
    return y0

@njit
def conv_bound(W, b, pad, stride, x0, eps, p_n):
    y0 = conv(W, x0, pad, stride)
    UB = np.zeros(y0.shape, dtype=np.float32)
    LB = np.zeros(y0.shape, dtype=np.float32)
    for k in range(W.shape[0]):
        if p_n == 105: # p == "i", q = 1
            dualnorm = np.sum(np.abs(W[k,:,:,:]))
        elif p_n == 1: # p = 1, q = i
            dualnorm = np.max(np.abs(W[k,:,:,:]))
        elif p_n == 2: # p = 2, q = 2
            dualnorm = np.sqrt(np.sum(W[k,:,:,:]**2))
        mid = y0[:,:,k]+b[k]
        UB[:,:,k] = mid+eps*dualnorm
        LB[:,:,k] = mid-eps*dualnorm
    return LB, UB

@njit
def conv_full(A, x, pad, stride):
    p_hl, p_hr, p_wl, p_wr = pad
    s_h, s_w = stride
    y = np.zeros((A.shape[0], A.shape[1], A.shape[2]), dtype=np.float32)
    for a in range(y.shape[0]):
        for b in range(y.shape[1]):
            for c in range(y.shape[2]):
                for i in range(A.shape[3]):
                    for j in range(A.shape[4]):
                        for k in range(A.shape[5]):
                            if 0<=s_h*a+i-p_hl<x.shape[0] and 0<=s_w*b+j-p_wl<x.shape[1]:
                                y[a,b,c] += A[a,b,c,i,j,k]*x[s_h*a+i-p_hl,s_w*b+j-p_wl,k]
    return y

@njit
def conv_bound_full(A, B, pad, stride, x0, eps, p_n):
    y0 = conv_full(A, x0, pad, stride)
    UB = np.zeros(y0.shape, dtype=np.float32)
    LB = np.zeros(y0.shape, dtype=np.float32)
    for a in range(y0.shape[0]):
        for b in range(y0.shape[1]):
            for c in range(y0.shape[2]):
                if p_n == 105: # p == "i", q = 1
                    dualnorm = np.sum(np.abs(A[a,b,c,:,:,:]))
                elif p_n == 1: # p = 1, q = i
                    dualnorm = np.max(np.abs(A[a,b,c,:,:,:]))
                elif p_n == 2: # p = 2, q = 2
                    dualnorm = np.sqrt(np.sum(A[a,b,c,:,:,:]**2))
                mid = y0[a,b,c]+B[a,b,c]
                UB[a,b,c] = mid+eps*dualnorm
                LB[a,b,c] = mid-eps*dualnorm
    return LB, UB

@njit
def upper_bound_conv(A, B, pad, stride, W, b, inner_pad, inner_stride, inner_shape, LB, UB, strategy_map_LB, strategy_map_UB, method):
    A_new = np.zeros((A.shape[0], A.shape[1], A.shape[2], inner_stride[0]*(A.shape[3]-1)+W.shape[1], inner_stride[1]*(A.shape[4]-1)+W.shape[2], W.shape[3]), dtype=np.float32)
    B_new = np.zeros(B.shape, dtype=np.float32)
    A_plus = np.maximum(A, 0)
    A_minus = np.minimum(A, 0)
    alpha_u, alpha_l, beta_u, beta_l = linear_bounds(LB, UB, strategy_map_LB, strategy_map_UB, method)
    assert A.shape[5] == W.shape[0]

    for x in range(A_new.shape[0]):
        for y in range(A_new.shape[1]):
            for t in range(A_new.shape[3]):
                for u in range(A_new.shape[4]):
                    if 0<=t+stride[0]*inner_stride[0]*x-inner_stride[0]*pad[0]-inner_pad[0]<inner_shape[0] and 0<=u+stride[1]*inner_stride[1]*y-inner_stride[1]*pad[2]-inner_pad[2]<inner_shape[1]:
                        for p in range(A.shape[3]):
                            for q in range(A.shape[4]):
                                if 0<=t-inner_stride[0]*p<W.shape[1] and 0<=u-inner_stride[1]*q<W.shape[2] and 0<=p+stride[0]*x-pad[0]<alpha_u.shape[0] and 0<=q+stride[1]*y-pad[2]<alpha_u.shape[1]:
                                    for z in range(A_new.shape[2]):
                                        for v in range(A_new.shape[5]):
                                            for r in range(W.shape[0]):
                                                A_new[x,y,z,t,u,v] += W[r,t-inner_stride[0]*p,u-inner_stride[1]*q,v]*alpha_u[p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],r]*A_plus[x,y,z,p,q,r]
                                                A_new[x,y,z,t,u,v] += W[r,t-inner_stride[0]*p,u-inner_stride[1]*q,v]*alpha_l[p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],r]*A_minus[x,y,z,p,q,r]
                                                
    B_new = conv_full(A_plus,alpha_u*b+beta_u,pad,stride) + conv_full(A_minus,alpha_l*b+beta_l,pad,stride)+B
    return A_new, B_new


@njit
def lower_bound_conv(A, B, pad, stride, W, b, inner_pad, inner_stride, inner_shape, LB, UB, strategy_map_LB, strategy_map_UB, method):
    A_new = np.zeros((A.shape[0], A.shape[1], A.shape[2], inner_stride[0]*(A.shape[3]-1)+W.shape[1], inner_stride[1]*(A.shape[4]-1)+W.shape[2], W.shape[3]), dtype=np.float32)
    B_new = np.zeros(B.shape, dtype=np.float32)
    A_plus = np.maximum(A, 0)
    A_minus = np.minimum(A, 0)
    alpha_u, alpha_l, beta_u, beta_l = linear_bounds(LB, UB, strategy_map_LB, strategy_map_UB, method)
    assert A.shape[5] == W.shape[0]
    for x in range(A_new.shape[0]):
        for y in range(A_new.shape[1]):
            for t in range(A_new.shape[3]):
                for u in range(A_new.shape[4]):
                    if 0<=t+stride[0]*inner_stride[0]*x-inner_stride[0]*pad[0]-inner_pad[0]<inner_shape[0] and 0<=u+stride[1]*inner_stride[1]*y-inner_stride[1]*pad[2]-inner_pad[2]<inner_shape[1]:
                        for p in range(A.shape[3]):
                            for q in range(A.shape[4]):
                                if 0<=t-inner_stride[0]*p<W.shape[1] and 0<=u-inner_stride[1]*q<W.shape[2] and 0<=p+stride[0]*x-pad[0]<alpha_u.shape[0] and 0<=q+stride[1]*y-pad[2]<alpha_u.shape[1]:
                                    for z in range(A_new.shape[2]):
                                        for v in range(A_new.shape[5]):
                                            for r in range(W.shape[0]):
                                                A_new[x,y,z,t,u,v] += W[r,t-inner_stride[0]*p,u-inner_stride[1]*q,v]*alpha_l[p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],r]*A_plus[x,y,z,p,q,r]
                                                A_new[x,y,z,t,u,v] += W[r,t-inner_stride[0]*p,u-inner_stride[1]*q,v]*alpha_u[p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],r]*A_minus[x,y,z,p,q,r]
    B_new = conv_full(A_plus,alpha_l*b+beta_l,pad,stride) + conv_full(A_minus,alpha_u*b+beta_u,pad,stride)+B
    return A_new, B_new


@njit
def pool_linear_bounds(LB, UB, pad, stride, pool_size):
    p_hl, p_hr, p_wl, p_wr = pad
    s_h, s_w = stride
    alpha_u = np.zeros((pool_size[0], pool_size[1], int((UB.shape[0]+p_hl+p_hr-pool_size[0])/s_h)+1, int((UB.shape[1]+p_wl+p_wr-pool_size[1])/s_w)+1, UB.shape[2]), dtype=np.float32)
    beta_u = np.zeros((int((UB.shape[0]+p_hl+p_hr-pool_size[0])/s_h)+1, int((UB.shape[1]+p_wl+p_wr-pool_size[1])/s_w)+1, UB.shape[2]), dtype=np.float32)
    alpha_l = np.zeros((pool_size[0], pool_size[1], int((LB.shape[0]+p_hl+p_hr-pool_size[0])/s_h)+1, int((LB.shape[1]+p_wl+p_wr-pool_size[1])/s_w)+1, LB.shape[2]), dtype=np.float32)
    beta_l = np.zeros((int((LB.shape[0]+p_hl+p_hr-pool_size[0])/s_h)+1, int((LB.shape[1]+p_wl+p_wr-pool_size[1])/s_w)+1, LB.shape[2]), dtype=np.float32)

    for x in range(alpha_u.shape[2]):
        for y in range(alpha_u.shape[3]):
            for r in range(alpha_u.shape[4]):
                cropped_LB = LB[s_h*x-p_hl:pool_size[0]+s_h*x-p_hl, s_w*y-p_wl:pool_size[1]+s_w*y-p_wl,r]
                cropped_UB = UB[s_h*x-p_hl:pool_size[0]+s_h*x-p_hl, s_w*y-p_wl:pool_size[1]+s_w*y-p_wl,r]

                max_LB = cropped_LB.max()
                idx = np.where(cropped_UB>=max_LB)
                u_s = np.zeros(len(idx[0]), dtype=np.float32)
                l_s = np.zeros(len(idx[0]), dtype=np.float32)
                gamma = np.inf
                for i in range(len(idx[0])):
                    l_s[i] = cropped_LB[idx[0][i],idx[1][i]]
                    u_s[i] = cropped_UB[idx[0][i],idx[1][i]]
                    if l_s[i] == u_s[i]:
                        gamma = l_s[i]

                if gamma == np.inf:
                    gamma = (np.sum(u_s/(u_s-l_s))-1)/np.sum(1/(u_s-l_s))
                    if gamma < np.max(l_s):
                        gamma = np.max(l_s)
                    elif gamma > np.min(u_s):
                        gamma = np.min(u_s)
                    weights = ((u_s-gamma)/(u_s-l_s)).astype(np.float32)
                else:
                    weights = np.zeros(len(idx[0]), dtype=np.float32)
                    w_partial_sum = 0
                    num_equal = 0
                    for i in range(len(idx[0])):
                        if l_s[i] != u_s[i]:
                            weights[i] = (u_s[i]-gamma)/(u_s[i]-l_s[i])
                            w_partial_sum += weights[i]
                        else:
                            num_equal += 1
                    gap = (1-w_partial_sum)/num_equal
                    if gap < 0.0:
                        gap = 0.0
                    elif gap > 1.0:
                        gap = 1.0
                    for i in range(len(idx[0])):
                        if l_s[i] == u_s[i]:
                            weights[i] = gap

                for i in range(len(idx[0])):
                    t = idx[0][i]
                    u = idx[1][i]
                    alpha_u[t,u,x,y,r] = weights[i]
                    alpha_l[t,u,x,y,r] = weights[i]
                beta_u[x,y,r] = gamma-np.dot(weights, l_s)
                growth_rate = np.sum(weights)
                if growth_rate <= 1:
                    beta_l[x,y,r] = np.min(l_s)*(1-growth_rate)
                else:
                    beta_l[x,y,r] = np.max(u_s)*(1-growth_rate)
    return alpha_u, alpha_l, beta_u, beta_l

@njit
def upper_bound_pool(A, B, pad, stride, pool_size, inner_pad, inner_stride, inner_shape, LB, UB):
    A_new = np.zeros((A.shape[0], A.shape[1], A.shape[2], inner_stride[0]*(A.shape[3]-1)+pool_size[0], inner_stride[1]*(A.shape[4]-1)+pool_size[1], A.shape[5]), dtype=np.float32)
    B_new = np.zeros(B.shape, dtype=np.float32)
    A_plus = np.maximum(A, 0)
    A_minus = np.minimum(A, 0)
    alpha_u, alpha_l, beta_u, beta_l = pool_linear_bounds(LB, UB, inner_pad, inner_stride, pool_size)

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
    B_new = conv_full(A_plus,beta_u,pad,stride) + conv_full(A_minus,beta_l,pad,stride)+B
    return A_new, B_new

@njit
def lower_bound_pool(A, B, pad, stride, pool_size, inner_pad, inner_stride, inner_shape, LB, UB):
    A_new = np.zeros((A.shape[0], A.shape[1], A.shape[2], inner_stride[0]*(A.shape[3]-1)+pool_size[0], inner_stride[1]*(A.shape[4]-1)+pool_size[1], A.shape[5]), dtype=np.float32)
    B_new = np.zeros(B.shape, dtype=np.float32)
    A_plus = np.maximum(A, 0)
    A_minus = np.minimum(A, 0)
    alpha_u, alpha_l, beta_u, beta_l = pool_linear_bounds(LB, UB, inner_pad, inner_stride, pool_size)

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
                                    A_new[x,y,:,t,u,:] += A_plus[x,y,:,p,q,:]*alpha_l[t-inner_stride[0]*p,u-inner_stride[1]*q,p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],:]
                                    A_new[x,y,:,t,u,:] += A_minus[x,y,:,p,q,:]*alpha_u[t-inner_stride[0]*p,u-inner_stride[1]*q,p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],:]
    B_new = conv_full(A_plus,beta_l,pad,stride) + conv_full(A_minus,beta_u,pad,stride)+B
    return A_new, B_new

@njit
def compute_bounds(weights, biases, out_shape, nlayer, x0, eps, p_n, strides, pads, LBs, UBs, strategy_map_LB, strategy_map_UB, method):
    # print('nlayer: ', nlayer)
    pad = (0,0,0,0)
    stride = (1,1)
    modified_LBs = LBs + (np.ones(out_shape, dtype=np.float32),)
    modified_UBs = UBs + (np.ones(out_shape, dtype=np.float32),)
    for i in range(nlayer-1, -1, -1):
        if not np.isnan(weights[i]).any(): #Conv
            if i == nlayer-1:
                A_u = weights[i].reshape((1, 1, weights[i].shape[0], weights[i].shape[1], weights[i].shape[2], weights[i].shape[3]))*np.ones((out_shape[0], out_shape[1], weights[i].shape[0], weights[i].shape[1], weights[i].shape[2], weights[i].shape[3]), dtype=np.float32)
                B_u = biases[i]*np.ones((out_shape[0], out_shape[1], out_shape[2]), dtype=np.float32)
                A_l = A_u.copy()
                B_l = B_u.copy()
            else:
                A_u, B_u = upper_bound_conv(A_u, B_u, pad, stride, weights[i], biases[i], pads[i], strides[i], modified_UBs[i].shape, modified_LBs[i+1], modified_UBs[i+1], strategy_map_LB[i+1], strategy_map_UB[i+1], method)
                A_l, B_l = lower_bound_conv(A_l, B_l, pad, stride, weights[i], biases[i], pads[i], strides[i], modified_LBs[i].shape, modified_LBs[i+1], modified_UBs[i+1], strategy_map_LB[i+1], strategy_map_UB[i+1], method) 
        else: #Pool
            if i == nlayer-1:
                A_u = np.eye(out_shape[2]).astype(np.float32).reshape((1,1,out_shape[2],1,1,out_shape[2]))*np.ones((out_shape[0], out_shape[1], out_shape[2], 1,1,out_shape[2]), dtype=np.float32)
                B_u = np.zeros(out_shape, dtype=np.float32)
                A_l = A_u.copy()
                B_l = B_u.copy()
            A_u, B_u = upper_bound_pool(A_u, B_u, pad, stride, weights[i].shape[1:], pads[i], strides[i], modified_UBs[i].shape, np.maximum(modified_LBs[i],0), np.maximum(modified_UBs[i],0))
            A_l, B_l = lower_bound_pool(A_l, B_l, pad, stride, weights[i].shape[1:], pads[i], strides[i], modified_LBs[i].shape, np.maximum(modified_LBs[i],0), np.maximum(modified_UBs[i],0))
        pad = (strides[i][0]*pad[0]+pads[i][0], strides[i][0]*pad[1]+pads[i][1], strides[i][1]*pad[2]+pads[i][2], strides[i][1]*pad[3]+pads[i][3])
        stride = (strides[i][0]*stride[0], strides[i][1]*stride[1])
    LUB, UUB = conv_bound_full(A_u, B_u, pad, stride, x0, eps, p_n)
    LLB, ULB = conv_bound_full(A_l, B_l, pad, stride, x0, eps, p_n)
    return LLB, ULB, LUB, UUB, A_u, A_l, B_u, B_l, pad, stride

def find_output_bounds(weights, biases, shapes, pads, strides, x0, eps, p_n, strategy_map_LB, strategy_map_UB, method='NeWise'):
    LB, UB = conv_bound(weights[0], biases[0], pads[0], strides[0], x0, eps, p_n)
    LBs = [x0-eps, LB]
    UBs = [x0+eps, UB]
    
    # printlog('--------------------1 layer -------------------')
    # for one in range(LB.shape[0]):
    #     for two in range(LB.shape[1]):
    #         for three in range(LB.shape[2]):
    #             # printlog("[{:.5f}, {:.5f}]".format(LB[one,two,three], UB[one,two,three]))
    #             printlog("{:.5f}".format(UB[one,two,three]-LB[one,two,three]))
    
    for i in range(2,len(weights)+1):
        # print('find_output_bounds ', i)
        LB, _, _, UB, A_u, A_l, B_u, B_l, pad, stride = compute_bounds(tuple(weights), tuple(biases), shapes[i], i, x0, eps, p_n, tuple(strides), tuple(pads), tuple(LBs), tuple(UBs), strategy_map_LB, strategy_map_UB, method)
        UBs.append(UB)
        LBs.append(LB)
        
        # printlog('--------------------' + str(i) + ' layer -------------------')
        # for one in range(LB.shape[0]):
        #     for two in range(LB.shape[1]):
        #         for three in range(LB.shape[2]):
        #             # printlog("[{:.5f}, {:.5f}]".format(LB[one,two,three], UB[one,two,three]))
        #             printlog("{:.5f}".format(UB[one,two,three]-LB[one,two,three]))

    # return LBs, UBs, A_u, A_l, B_u, B_l, pad, stride
    return LBs[-1], UBs[-1], A_u, A_l, B_u, B_l, pad, stride

def warmup(model, x, eps_0, p_n, fn, strategy_map_LB, strategy_map_UB):
    
    print('Warming up...')
    weights = model.weights[:-1]
    biases = model.biases[:-1]
    shapes = model.shapes[:-1]
    W, b, s = model.weights[-1], model.biases[-1], model.shapes[-1]
    last_weight = np.ascontiguousarray((W[0,:,:,:]).reshape([1]+list(W.shape[1:])),dtype=np.float32)
    weights.append(last_weight)
    biases.append(np.asarray([b[0]]))
    shapes.append((1,1,1))
    print('enter fn...')
    fn(weights, biases, shapes, model.pads, model.strides, x, eps_0, p_n, strategy_map_LB, strategy_map_UB)
    
ts = time.time()
timestr = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')
#Prints to log file
def printlog(s):
    print(s)
    print(s, file=open("logs/cnn_bounds_full_core_with_LP_"+timestr+".txt", "a"))

def run_verified_robustness_ratio_core(file_name, n_samples, p_n, q_n, data_from_local=True, method='NeWise', sample_num=0, cnn_cert_model=False, activation = 'sigmoid', mnist=False, cifar=False, fashion_mnist=False, gtsrb=False, step_tmp = 0.45, eps=0.002, eran_fnn=False):
    np.random.seed(1215)
    random.seed(1215)
    if activation == 'atan':
        keras_model = load_model(file_name, custom_objects={'atan': tf.atan})
    else:
        keras_model = load_model(file_name, custom_objects={'fn':fn, 'tf':tf})

    if cifar:
        model = CNNModel(keras_model, inp_shape = (32,32,3))
    elif gtsrb:
        print('gtsrb')
        model = CNNModel(keras_model, inp_shape = (48,48,3))
    else:
        model = CNNModel(keras_model)
    print('--------abstracted model-----------')
    
    global linear_bounds
    if activation == 'sigmoid':
        linear_bounds = sigmoid_linear_bounds
    elif activation == 'tanh':
        linear_bounds = tanh_linear_bounds
    elif activation == 'atan':
        linear_bounds = atan_linear_bounds
    
    upper_bound_conv.recompile()
    lower_bound_conv.recompile()
    compute_bounds.recompile()

    dataset = ''
    
    if cifar:
        dataset = 'cifar10'
        inputs, targets, true_labels, true_ids, img_info = generate_data('cifar10', samples=n_samples, data_from_local=data_from_local, targeted=True, random_and_least_likely = True, target_type = 0b0010, predictor=model.model.predict, start=0, cnn_cert_model=cnn_cert_model, eran_fnn=eran_fnn)
    elif fashion_mnist:
        dataset = 'fashion_mnist'
        inputs, targets, true_labels, true_ids, img_info = generate_data('fashion_mnist', samples=n_samples, data_from_local=data_from_local, targeted=True, random_and_least_likely = True, target_type = 0b0010, predictor=model.model.predict, start=0, cnn_cert_model=cnn_cert_model, eran_fnn=eran_fnn)
    else:
        dataset = 'mnist'
        inputs, targets, true_labels, true_ids, img_info = generate_data('mnist', samples=n_samples, data_from_local=data_from_local, targeted=True, random_and_least_likely = True, target_type = 0b0010, predictor=model.model.predict, start=0, cnn_cert_model=cnn_cert_model, eran_fnn=eran_fnn)
        
    #0b01111 <- all
    #0b0010 <- random
    #0b0001 <- top2 
    #0b0100 <- least
        
    print('----------generated data---------')

    printlog('===========================================')
    printlog("model name = {}".format(file_name))
    
    # printlog('====================' + method + '=======================')
    
    total_images = 0
    # sample_num = 0
    
    strategy_map_LB = forward_propagation(inputs[0].astype(np.float32), model.weights, model.biases, model.shapes, model.pads, model.strides, 'sigmoid')
    strategy_map_UB = strategy_map_LB
    warmup(model, inputs[0].astype(np.float32), eps, p_n, find_output_bounds, strategy_map_LB, strategy_map_UB)
    
    sampling_total_time = 0
    verify_total_time = 0
    GD_total_time = 0

    robust = 0
    unknown = 0
    not_robust = 0
    for i in range(len(inputs)):
        # printlog('--- ' + method + ' relaxation: Computing eps for input image ' + str(i)+ '---')
        predict_label = np.argmax(true_labels[i])
        target_label = np.argmax(targets[i])
        

        if method == 'gradient_descent':
            feed_x_0 = tf.convert_to_tensor(inputs[i:i+1].astype(np.float32))
            layer_num = len(keras_model.layers)
            GRAD_ALL = []
            
            GD_start_time = time.time()
            
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(feed_x_0)
                x = feed_x_0
                flag = 0
                for cur_layer, layer in enumerate(keras_model.layers):
                    weights = layer.get_weights()
                    GRAD = []
                    if type(keras_model.layers[cur_layer]) == Conv2D:
                        if flag == 1:
                            if activation == 'sigmoid':
                                x = tf.nn.sigmoid(x)
                            elif activation == 'tanh':
                                x = tf.nn.tanh(x)
                            elif activation == 'atan':
                                x = tf.atan(x)
                        elif flag == 0:
                            flag = 1
                        if len(weights) == 1:
                            W = weights
                        else:
                            W, b = weights
                        padding = layer.get_config()['padding']
                        if padding == 'same':
                            padding = 'SAME'
                        elif padding == 'valid':
                            padding = 'VALID'
                        else:
                            padding = 'EXPLICIT'
                        stride = layer.get_config()['strides']
                        x = tf.nn.conv2d(x, W, strides = stride, padding = padding) + b
                        cur_shape = keras_model.layers[cur_layer].output_shape
                        for a in range(cur_shape[1]):
                            GRAD_cur_1 = []
                            for b in range(cur_shape[2]):
                                GRAD_cur_2 = []
                                for c in range(cur_shape[3]):
                                    grad_cur_neuron = tape.gradient(target = x[0][a][b][c], sources = feed_x_0)
                                    # grad_cur_neuron = grad_cur_neuron.numpy()
                                    GRAD_cur_2.append(tf.sign(grad_cur_neuron))
                                GRAD_cur_1.append(GRAD_cur_2)
                            GRAD.append(GRAD_cur_1) 

                    elif type(keras_model.layers[cur_layer]) == Dense:
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
                        W, b = weights
                        x = tf.reshape(x, [1, W.shape[0]])
                        x = tf.matmul(x, W) + b
                        for cur_neuron in range(num_neurons):
                            grad_cur_neuron = tape.gradient(target=x[0][cur_neuron], sources=feed_x_0)
                            # grad_cur_neuron = grad_cur_neuron.numpy()
                            GRAD.append(tf.sign(grad_cur_neuron))
                    GRAD_ALL.append(GRAD)
            del tape
            
            GD_end_time = time.time() - GD_start_time
            print("sign of GD consumes: ", GD_end_time)
            GD_total_time += GD_end_time

    
        if method == 'gradient_descent':
            GD_start_time = time.time()

            strategy_map_LB = []
            strategy_map_UB = []
            cur_eps = eps

            strategy_map_LB.append(inputs[i].astype(np.float32) - cur_eps)
            strategy_map_UB.append(inputs[i].astype(np.float32) + cur_eps)
            
            tmp_cnt = 0
            for cur_layer in range(layer_num):
                if type(keras_model.layers[cur_layer]) == Conv2D:
                    tmp_cnt += 1
                    cur_shape = keras_model.layers[cur_layer].output_shape
                    ub_ans = np.zeros((cur_shape[1], cur_shape[2], cur_shape[3]), dtype=np.float32)
                    lb_ans = np.zeros((cur_shape[1], cur_shape[2], cur_shape[3]), dtype=np.float32)
                    for a in range(cur_shape[1]):
                        for b in range(cur_shape[2]):
                            for c in range(cur_shape[3]):
                                grad_cur_neuron = GRAD_ALL[cur_layer][a][b][c].numpy()

                                x = copy.deepcopy(feed_x_0)
                                x -= cur_eps * grad_cur_neuron * step_tmp
                                x = np.clip(x, feed_x_0 - cur_eps, feed_x_0 + cur_eps) 
                                lb_ans[a][b][c] = forward_propagation(x[0], model.weights, model.biases, model.shapes, model.pads, model.strides, activation)[tmp_cnt][a][b][c]

                                x = copy.deepcopy(feed_x_0)
                                x += cur_eps * grad_cur_neuron * step_tmp
                                x = np.clip(x, feed_x_0 - cur_eps, feed_x_0 + cur_eps) 
                                ub_ans[a][b][c] = forward_propagation(x[0], model.weights, model.biases, model.shapes, model.pads, model.strides, activation)[tmp_cnt][a][b][c]
                                                
                    ub_ans_copy = ub_ans.copy().astype(np.float32)
                    lb_ans_copy = lb_ans.copy().astype(np.float32)
                    strategy_map_UB.append(ub_ans_copy)
                    strategy_map_LB.append(lb_ans_copy)

                elif type(keras_model.layers[cur_layer]) == Dense:
                    tmp_cnt += 1
                    
                    num_neurons = keras_model.layers[cur_layer].output_shape[-1]
                    shape = (1, 1, num_neurons)
                    ub_ans = np.zeros(shape, dtype=np.float32)
                    lb_ans = np.zeros(shape, dtype=np.float32)
                    for cur_neuron in range(num_neurons):
                        grad_cur_neuron = GRAD_ALL[cur_layer][cur_neuron].numpy()

                        x = copy.deepcopy(feed_x_0)
                        x -= cur_eps * grad_cur_neuron * step_tmp
                        x = np.clip(x, feed_x_0 - cur_eps, feed_x_0 + cur_eps) 
                        lb_ans[0][0][cur_neuron] = forward_propagation(x[0], model.weights, model.biases, model.shapes, model.pads, model.strides, activation)[tmp_cnt][0][0][cur_neuron]

                        x = copy.deepcopy(feed_x_0)
                        x += cur_eps * grad_cur_neuron * step_tmp
                        x = np.clip(x, feed_x_0 - cur_eps, feed_x_0 + cur_eps) 
                        ub_ans[0][0][cur_neuron] = forward_propagation(x[0], model.weights, model.biases, model.shapes, model.pads, model.strides, activation)[tmp_cnt][0][0][cur_neuron]
                    
                    ub_ans_copy = ub_ans.copy().astype(np.float32)
                    lb_ans_copy = lb_ans.copy().astype(np.float32)
                    strategy_map_UB.append(ub_ans_copy)
                    strategy_map_LB.append(lb_ans_copy)

            GD_end_time = time.time() - GD_start_time
            GD_total_time += GD_end_time


        if method == 'guided_by_median':
            sampling_start_time = time.time()
            
            # 1. 产生 sample_num 个随机样本，使用 numpy.ndarray 结构存储，shape: (sample_num, input_shape)
            # sample_num = 300
            samples_from_ith_image_shape = (sample_num, inputs[i].shape[0], inputs[i].shape[1], inputs[i].shape[2])
            samples_from_ith_image = np.random.uniform(inputs[i] - eps, inputs[i] + eps, samples_from_ith_image_shape).astype(np.float32)
            
            # 2. 对这 sample_num 个随机样本进行前向传播，创建 sample_results 用于存储 sample_num 个采样点在每个节点的取值
            # 因为每一层的 shape 都不一样，所以对于每个采样点在每个节点的取值只能用 list 进行存储，
            # 进而 sample_num 个采样点的总结果也存储于 list 结构中，每个采样点在每一层的取值结果用 numpy.array 存储
            samples_from_ith_image_results = []
            
            for sample_index in range(sample_num):
                samples_from_ith_image_results.append(forward_propagation(samples_from_ith_image[sample_index], model.weights, model.biases, model.shapes, model.pads, model.strides, activation))
            
            # 3. 使用 numpy 的统计函数得到每个节点上的中位数
            strategy_map_LB = []
            layer_num = len(samples_from_ith_image_results[0])
            
            printlog('====================== Sampling ======================')
                            
            for nn_layer in range(layer_num):
                t_shape = (sample_num, samples_from_ith_image_results[0][nn_layer].shape[0], samples_from_ith_image_results[0][nn_layer].shape[1], samples_from_ith_image_results[0][nn_layer].shape[2])
                t = np.zeros(t_shape, dtype=np.float32)
                for index in range(sample_num):
                    t[index] = samples_from_ith_image_results[index][nn_layer]
                median_ans = np.median(t, axis=0)
                ans = median_ans.copy().astype(np.float32)
                strategy_map.append(ans)
                printlog('--------------------'+str(nn_layer)+' layer -------------------')
                minimum = np.amin(t, axis=0)
                maximum = np.amax(t, axis=0)
                for one in range(minimum.shape[0]):
                    for two in range(minimum.shape[1]):
                        for three in range(minimum.shape[2]):
                            printlog("[{:.5f}, {:.5f}]".format(minimum[one,two,three], maximum[one,two,three]))
                            
            strategy_map_UB = strategy_map_LB
            printlog('====================== Sampling End ======================')
        
            sampling_end_time = time.time() - sampling_start_time
            sampling_total_time += sampling_end_time
            # print('sampling time: ', sampling_end_time)
        
        elif method == 'guided_by_endpoint':
            sampling_start_time = time.time()
            
            # 1. 产生 sample_num 个随机样本，使用 numpy.ndarray 结构存储，shape: (sample_num, input_shape)
            # sample_num = 300
            samples_from_ith_image_shape = (sample_num, inputs[i].shape[0], inputs[i].shape[1], inputs[i].shape[2])
            samples_from_ith_image = np.random.uniform(inputs[i] - eps, inputs[i] + eps, samples_from_ith_image_shape).astype(np.float32)
            
            # 2. 对这 sample_num 个随机样本进行前向传播，创建 sample_results 用于存储 sample_num 个采样点在每个节点的取值
            # 因为每一层的 shape 都不一样，所以对于每个采样点在每个节点的取值只能用 list 进行存储，
            # 进而 sample_num 个采样点的总结果也存储于 list 结构中，每个采样点在每一层的取值结果用 numpy.array 存储
            samples_from_ith_image_results = []
            
            for sample_index in range(sample_num):
                samples_from_ith_image_results.append(forward_propagation(samples_from_ith_image[sample_index], model.weights, model.biases, model.shapes, model.pads, model.strides, activation))
            
            # 3. 使用 numpy 的统计函数得到每个节点上的中位数
            strategy_map_LB = []
            strategy_map_UB = []
            layer_num = len(samples_from_ith_image_results[0])
                            
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
        
        # printlog('====================== Overapproximation ======================')
        
        verify_start_time = time.time()
        if method == 'gradient_descent':
            method_core = 'guided_by_endpoint'
        else:
            method_core = method
        
        # LB_total, UB_total, _, _, _, _, _, _ = find_output_bounds(model.weights, model.biases, model.shapes, model.pads, model.strides, inputs[i].astype(np.float32), eps, p_n, strategy_map_LB, strategy_map_UB, method_core)
        LB_total, UB_total, _, _, _, _, _, _ = find_output_bounds(model.weights, model.biases, model.shapes, model.pads, model.strides, inputs[i].astype(np.float32), eps, p_n, strategy_map_LB, strategy_map_UB, method_core)

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
            else:                                          # 新
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
            # LB, UB, A_u, A_l, B_u, B_l, pad, stride = find_output_bounds(weights, biases, shapes, model.pads, model.strides, inputs[i].astype(np.float32), eps, p_n, strategy_map_LB, strategy_map_UB, method_core)
            
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
        # elif not find_conter_example:                          新
        #     unknown += 1
        #     print(f"Image {i} is unknown, unknown number = {unknown}")
        verify_total_time += time.time() - verify_start_time
        

    
    # aver_time = (time.time()-NeWise_start_time)/len(inputs)
    aver_GD_time = GD_total_time / len(inputs)
    aver_sample_time = sampling_total_time / len(inputs)
    aver_verify_time = verify_total_time / len(inputs)
    # printlog("[L0] method = {}-{}, total images = {}, avg robustness = {:.5f}, avg verify runtime = {:.2f}, avg sample runtime = {:.2f}, avg GD runtime = {:.2f}, sample num = {}".format(method, activation, len(inputs),eps,aver_verify_time, aver_sample_time, aver_GD_time, sample_num))
    print("### Summary: eps={:.5f}, wrong_classification={}, robust_images={}, robust_rate={:.5f}, unknown={}, not_robust={}".format(eps, n_samples - len(inputs), robust, float(robust)/float(len(inputs)), unknown, not_robust))
    print(f"Time: aver_GD_time = {aver_GD_time}, aver_sample_time = {aver_sample_time}, aver_verify_time = {aver_verify_time}")




# 主方法
def run_certified_bounds_core(file_name, n_samples, p_n, q_n, data_from_local=True, method='NeWise', sample_num=0, cnn_cert_model=False, activation = 'sigmoid', mnist=False, cifar=False, fashion_mnist=False, gtsrb=False, step_tmp = 0.45, eran_fnn=False):
    """
    file_name: 预训练模型名
    n_samples: batch_size
    p_n: 105
    q_n: 1  未使用
    data_from_local: 是否使用预下载数据
    method: 要使用的近似方法, NeWise等
    sample_num: 基于采样的下近似方法，一张图片在扰动区间中的采样数目
    cnn_cert_model: 
    activation: 激活函数
    mnist: 是否使用mnist数据集验证
    cifar: 是否使用cifar数据集验证
    fashion_mnist: 是否使用fashion_mnist数据集验证
    gtsrb: 是否使用gtsrb数据集验证
    step_tmp: 基于梯度的下近似方法的步长
    """


    np.random.seed(1215)
    random.seed(1215)
    # custom_objects: 将名称（字符串）映射到自定义类的可选字典  或反序列化期间要考虑的函数。
    if activation == 'atan':
        keras_model = load_model(file_name, custom_objects={'atan': tf.atan})
    else:
        keras_model = load_model(file_name, custom_objects={'fn':fn, 'tf':tf})

    if cifar:
        model = CNNModel(keras_model, inp_shape = (32,32,3))
    elif gtsrb:
        print('gtsrb')
        model = CNNModel(keras_model, inp_shape = (48,48,3))
    else:
        model = CNNModel(keras_model)

    print('--------abstracted model-----------')
    global linear_bounds
    if activation == 'sigmoid':
        linear_bounds = sigmoid_linear_bounds
    elif activation == 'tanh':
        linear_bounds = tanh_linear_bounds
    elif activation == 'atan':
        linear_bounds = atan_linear_bounds
    
    upper_bound_conv.recompile()
    lower_bound_conv.recompile()
    compute_bounds.recompile()

    dataset = ''
    
    # inputs, targets, true_labels, true_ids分别是输入图像，np.eye()，正确标签，图片位置序号
    if cifar:
        dataset = 'cifar10'
        inputs, targets, true_labels, true_ids, img_info = generate_data('cifar10', samples=n_samples, data_from_local=data_from_local, targeted=True, random_and_least_likely = True, target_type = 0b0010, predictor=model.model.predict, start=0, cnn_cert_model=cnn_cert_model, eran_fnn=eran_fnn)
    elif fashion_mnist:
        dataset = 'fashion_mnist'
        inputs, targets, true_labels, true_ids, img_info = generate_data('fashion_mnist', samples=n_samples, data_from_local=data_from_local, targeted=True, random_and_least_likely = True, target_type = 0b0010, predictor=model.model.predict, start=0, cnn_cert_model=cnn_cert_model, eran_fnn=eran_fnn)
    else:
        dataset = 'mnist'
        inputs, targets, true_labels, true_ids, img_info = generate_data('mnist', samples=n_samples, data_from_local=data_from_local, targeted=True, random_and_least_likely = True, target_type = 0b0010, predictor=model.model.predict, start=0, cnn_cert_model=cnn_cert_model, eran_fnn=eran_fnn)
    


    #0b01111 <- all
    #0b0010 <- random
    #0b0001 <- top2 
    #0b0100 <- least
        
    print('----------generated data---------')

    printlog('===========================================')
    printlog("model name = {}".format(file_name))
    
    # printlog('====================' + method + '=======================')
    
    total_images = 0
    steps = 15  # 超参数，更新多少步
    eps_0 = 0.05  # 初始的置信下界
    summation = 0  # 置信下界的和
    eps_total = np.zeros((len(inputs)), dtype=np.float32)  # 每个验证图像的置信下界
    # sample_num = 0
    
    # 精确结果，上界和下界相同
    strategy_map_LB = forward_propagation(inputs[0].astype(np.float32), model.weights, model.biases, model.shapes, model.pads, model.strides, 'sigmoid')
    strategy_map_UB = strategy_map_LB
    warmup(model, inputs[0].astype(np.float32), eps_0, p_n, find_output_bounds, strategy_map_LB, strategy_map_UB)
    
    sampling_total_time = 0
    verify_total_time = 0
    GD_total_time = 0

    for i in range(len(inputs)):
        # printlog('--- ' + method + ' relaxation: Computing eps for input image ' + str(i)+ '---')
        predict_label = np.argmax(true_labels[i])
        target_label = np.argmax(targets[i])

        
        #Perform binary search
        log_eps = np.log(eps_0)
        log_eps_min = -np.inf
        log_eps_max = np.inf

        # 这里计算的是每个神经元的值相对于输入的梯度
        if method == 'gradient_descent':
            feed_x_0 = tf.convert_to_tensor(inputs[i:i+1].astype(np.float32))
            layer_num = len(keras_model.layers)
            GRAD_ALL = []
            
            GD_start_time = time.time()
            
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(feed_x_0)
                x = feed_x_0
                flag = 0
                for cur_layer, layer in enumerate(keras_model.layers):
                    weights = layer.get_weights()
                    GRAD = []
                    if type(keras_model.layers[cur_layer]) == Conv2D:
                        if flag == 1:
                            if activation == 'sigmoid':
                                x = tf.nn.sigmoid(x)
                            elif activation == 'tanh':
                                x = tf.nn.tanh(x)
                            elif activation == 'atan':
                                x = tf.atan(x)
                        elif flag == 0:
                            flag = 1
                        if len(weights) == 1:
                            W = weights
                        else:
                            W, b = weights
                        padding = layer.get_config()['padding']
                        if padding == 'same':
                            padding = 'SAME'
                        elif padding == 'valid':
                            padding = 'VALID'
                        else:
                            padding = 'EXPLICIT'
                        stride = layer.get_config()['strides']
                        x = tf.nn.conv2d(x, W, strides = stride, padding = padding) + b
                        cur_shape = keras_model.layers[cur_layer].output_shape
                        for a in range(cur_shape[1]):
                            GRAD_cur_1 = []
                            for b in range(cur_shape[2]):
                                GRAD_cur_2 = []
                                for c in range(cur_shape[3]):
                                    grad_cur_neuron = tape.gradient(target = x[0][a][b][c], sources = feed_x_0)
                                    # grad_cur_neuron = grad_cur_neuron.numpy()
                                    GRAD_cur_2.append(tf.sign(grad_cur_neuron))
                                GRAD_cur_1.append(GRAD_cur_2)
                            GRAD.append(GRAD_cur_1) 

                    elif type(keras_model.layers[cur_layer]) == Dense:
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
                        W, b = weights
                        x = tf.reshape(x, [1, W.shape[0]])
                        x = tf.matmul(x, W) + b
                        for cur_neuron in range(num_neurons):
                            grad_cur_neuron = tape.gradient(target=x[0][cur_neuron], sources=feed_x_0)
                            # grad_cur_neuron = grad_cur_neuron.numpy()
                            GRAD.append(tf.sign(grad_cur_neuron))
                    GRAD_ALL.append(GRAD)
            del tape
            
            GD_end_time = time.time() - GD_start_time
            print("sign of GD consumes: ", GD_end_time)
            GD_total_time += GD_end_time

        for j in range(steps):  # 15步
            if method == 'gradient_descent':  # 梯度指导的更新
                # 梯度方法的主要思想就是一句目标函数的梯度计算两个有效的采样，来分别最小化和最大化目标函数的输出值
                GD_start_time = time.time()

                strategy_map_LB = []
                strategy_map_UB = []
                cur_eps = np.exp(log_eps)

                strategy_map_LB.append(inputs[i].astype(np.float32) - cur_eps)
                strategy_map_UB.append(inputs[i].astype(np.float32) + cur_eps)
                
                tmp_cnt = 0
                for cur_layer in range(layer_num):
                    if type(keras_model.layers[cur_layer]) == Conv2D:  # 卷积层
                        tmp_cnt += 1
                        cur_shape = keras_model.layers[cur_layer].output_shape
                        ub_ans = np.zeros((cur_shape[1], cur_shape[2], cur_shape[3]), dtype=np.float32)
                        lb_ans = np.zeros((cur_shape[1], cur_shape[2], cur_shape[3]), dtype=np.float32)
                        for a in range(cur_shape[1]):
                            for b in range(cur_shape[2]):
                                for c in range(cur_shape[3]):
                                    grad_cur_neuron = GRAD_ALL[cur_layer][a][b][c].numpy()

                                    x = copy.deepcopy(feed_x_0)
                                    x -= cur_eps * grad_cur_neuron * step_tmp  # 更新x，因为算的是相对于输入的梯度
                                    x = np.clip(x, feed_x_0 - cur_eps, feed_x_0 + cur_eps) 
                                    # 使用新的x算下界
                                    lb_ans[a][b][c] = forward_propagation(x[0], model.weights, model.biases, model.shapes, model.pads, model.strides, activation)[tmp_cnt][a][b][c]

                                    x = copy.deepcopy(feed_x_0)
                                    x += cur_eps * grad_cur_neuron * step_tmp
                                    x = np.clip(x, feed_x_0 - cur_eps, feed_x_0 + cur_eps) 
                                    ub_ans[a][b][c] = forward_propagation(x[0], model.weights, model.biases, model.shapes, model.pads, model.strides, activation)[tmp_cnt][a][b][c]
                                                    
                        ub_ans_copy = ub_ans.copy().astype(np.float32)
                        lb_ans_copy = lb_ans.copy().astype(np.float32)
                        strategy_map_UB.append(ub_ans_copy)
                        strategy_map_LB.append(lb_ans_copy)

                    elif type(keras_model.layers[cur_layer]) == Dense:
                        tmp_cnt += 1
                        
                        num_neurons = keras_model.layers[cur_layer].output_shape[-1]
                        shape = (1, 1, num_neurons)
                        ub_ans = np.zeros(shape, dtype=np.float32)
                        lb_ans = np.zeros(shape, dtype=np.float32)
                        for cur_neuron in range(num_neurons):
                            grad_cur_neuron = GRAD_ALL[cur_layer][cur_neuron].numpy()

                            x = copy.deepcopy(feed_x_0)
                            x -= cur_eps * grad_cur_neuron * step_tmp
                            x = np.clip(x, feed_x_0 - cur_eps, feed_x_0 + cur_eps) 
                            lb_ans[0][0][cur_neuron] = forward_propagation(x[0], model.weights, model.biases, model.shapes, model.pads, model.strides, activation)[tmp_cnt][0][0][cur_neuron]

                            x = copy.deepcopy(feed_x_0)
                            x += cur_eps * grad_cur_neuron * step_tmp
                            x = np.clip(x, feed_x_0 - cur_eps, feed_x_0 + cur_eps) 
                            ub_ans[0][0][cur_neuron] = forward_propagation(x[0], model.weights, model.biases, model.shapes, model.pads, model.strides, activation)[tmp_cnt][0][0][cur_neuron]
                        
                        ub_ans_copy = ub_ans.copy().astype(np.float32)
                        lb_ans_copy = lb_ans.copy().astype(np.float32)
                        strategy_map_UB.append(ub_ans_copy)
                        strategy_map_LB.append(lb_ans_copy)

                GD_end_time = time.time() - GD_start_time
                GD_total_time += GD_end_time


            if method == 'guided_by_median':  # 采样法，中点导向
                sampling_start_time = time.time()
                
                # 1. 产生 sample_num 个随机样本，使用 numpy.ndarray 结构存储，shape: (sample_num, input_shape)
                # sample_num = 300
                samples_from_ith_image_shape = (sample_num, inputs[i].shape[0], inputs[i].shape[1], inputs[i].shape[2])
                samples_from_ith_image = np.random.uniform(inputs[i] - np.exp(log_eps), inputs[i] + np.exp(log_eps), samples_from_ith_image_shape).astype(np.float32)
                
                # 2. 对这 sample_num 个随机样本进行前向传播，创建 sample_results 用于存储 sample_num 个采样点在每个节点的取值
                # 因为每一层的 shape 都不一样，所以对于每个采样点在每个节点的取值只能用 list 进行存储，
                # 进而 sample_num 个采样点的总结果也存储于 list 结构中，每个采样点在每一层的取值结果用 numpy.array 存储
                samples_from_ith_image_results = []
                
                for sample_index in range(sample_num):
                    samples_from_ith_image_results.append(forward_propagation(samples_from_ith_image[sample_index], model.weights, model.biases, model.shapes, model.pads, model.strides, activation))
                
                # 3. 使用 numpy 的统计函数得到每个节点上的中位数
                strategy_map_LB = []
                layer_num = len(samples_from_ith_image_results[0])
                
                printlog('====================== Sampling ======================')
                                
                for nn_layer in range(layer_num):
                    t_shape = (sample_num, samples_from_ith_image_results[0][nn_layer].shape[0], samples_from_ith_image_results[0][nn_layer].shape[1], samples_from_ith_image_results[0][nn_layer].shape[2])
                    t = np.zeros(t_shape, dtype=np.float32)
                    for index in range(sample_num):
                        t[index] = samples_from_ith_image_results[index][nn_layer]
                    median_ans = np.median(t, axis=0)
                    ans = median_ans.copy().astype(np.float32)
                    strategy_map_LB.append(ans)
                    printlog('--------------------'+str(nn_layer)+' layer -------------------')
                    minimum = np.amin(t, axis=0)
                    maximum = np.amax(t, axis=0)
                    for one in range(minimum.shape[0]):
                        for two in range(minimum.shape[1]):
                            for three in range(minimum.shape[2]):
                                printlog("[{:.5f}, {:.5f}]".format(minimum[one,two,three], maximum[one,two,three]))
                                
                strategy_map_UB = strategy_map_LB
                printlog('====================== Sampling End ======================')
            
                sampling_end_time = time.time() - sampling_start_time
                sampling_total_time += sampling_end_time
                # print('sampling time: ', sampling_end_time)
            
            elif method == 'guided_by_endpoint':  # 采样法，截止点导向
                sampling_start_time = time.time()
                
                # 1. 产生 sample_num 个随机样本，使用 numpy.ndarray 结构存储，shape: (sample_num, input_shape)
                # sample_num = 300
                samples_from_ith_image_shape = (sample_num, inputs[i].shape[0], inputs[i].shape[1], inputs[i].shape[2])
                samples_from_ith_image = np.random.uniform(inputs[i] - np.exp(log_eps), inputs[i] + np.exp(log_eps), samples_from_ith_image_shape).astype(np.float32)
                
                # 2. 对这 sample_num 个随机样本进行前向传播，创建 sample_results 用于存储 sample_num 个采样点在每个节点的取值
                # 因为每一层的 shape 都不一样，所以对于每个采样点在每个节点的取值只能用 list 进行存储，
                # 进而 sample_num 个采样点的总结果也存储于 list 结构中，每个采样点在每一层的取值结果用 numpy.array 存储
                samples_from_ith_image_results = []
                
                for sample_index in range(sample_num):
                    samples_from_ith_image_results.append(forward_propagation(samples_from_ith_image[sample_index], model.weights, model.biases, model.shapes, model.pads, model.strides, activation))
                
                # 3. 使用 numpy 的统计函数得到每个节点上的中位数
                strategy_map_LB = []
                strategy_map_UB = []
                layer_num = len(samples_from_ith_image_results[0])
                                
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
            
            # printlog('====================== Overapproximation ======================')
            
            # 开始验证
            verify_start_time = time.time()
            if method == 'gradient_descent':
                method_core = 'guided_by_endpoint'
            else:
                method_core = method
            # 获取输出上下界
            LB_total, UB_total, _, _, _, _, _, _ = find_output_bounds(model.weights, model.biases, model.shapes, model.pads, model.strides, inputs[i].astype(np.float32), np.exp(log_eps), p_n, strategy_map_LB, strategy_map_UB, method_core)
            # 预测标签值下界和其他标签值上界的距离
            distance_bt_pre_tar = LB_total[0][0][predict_label] - UB_total[0][0][target_label]
            print("Step {}, eps = {:.5f}, f_c_min - f_t_max = {:.6s}".format(j,np.exp(log_eps),str(distance_bt_pre_tar)))
            
            # print("Step {}, eps = {:.5f}, {:.6s} <= f_c - f_t <= {:.6s}".format(j,np.exp(log_eps),str(np.squeeze(LB)),str(np.squeeze(UB))))
            if distance_bt_pre_tar > 0:  # 当前验证鲁棒，Increase eps
                log_eps_min = log_eps
                log_eps = np.minimum(log_eps+1, (log_eps_max+log_eps_min)/2)
            else: # 当前验证不鲁棒，Decrease eps
                log_eps_max = log_eps
                log_eps = np.maximum(log_eps-1, (log_eps_max+log_eps_min)/2)
            verify_end_time = time.time() - verify_start_time
            verify_total_time += verify_end_time
            # print('verify time: ', verify_end_time)
        
        print("method = {}-{}, model = {}, image no = {}, true_id = {}, target_label = {}, true_label = {}, robustness = {:.5f}".format(method, activation,file_name, i, true_ids[i],target_label,predict_label,np.exp(log_eps_min)))
        summation += np.exp(log_eps_min)
        eps_total[i] = np.exp(log_eps_min)
    
    eps_avg = summation/len(inputs)
    # aver_time = (time.time()-NeWise_start_time)/len(inputs)
    aver_GD_time = GD_total_time / len(inputs)
    aver_sample_time = sampling_total_time / len(inputs)
    aver_verify_time = verify_total_time / len(inputs)
    # printlog("[L0] method = {}-{}, model = {}, total images = {}, avg robustness = {:.5f}, avg runtime = {:.2f}".format(method, activation,file_name,len(inputs),eps_avg,aver_time))
    printlog("[L0] method = {}-{}, total images = {}, avg robustness = {:.5f}, avg verify runtime = {:.2f}, avg sample runtime = {:.2f}, avg GD runtime = {:.2f}, sample num = {}, step = {}".format(method, activation, len(inputs),eps_avg,aver_verify_time, aver_sample_time, aver_GD_time, sample_num, steps))
    printlog("[L0] method = {}-{}, robustness: mean = {:.5f}, std = {:.5f}, var = {:.5f}, max = {:.5f}, min = {:.5f}".format(method, activation, np.mean(eps_total), np.std(eps_total), np.var(eps_total), np.amax(eps_total), np.amin(eps_total)))
    




