import gzip
import numpy as np

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Add, Lambda, Conv2D, AveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.constraints import non_neg, max_norm, min_max_norm
from tensorflow.keras.initializers import Constant

from tensorflow.python.keras.datasets import fashion_mnist
from tensorflow.python.ops import nn
import tensorflow as tf
import random
import os
import glob

from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.keras.models import load_model
import h5py

def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)
    
def get_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    x_train /= 255.
    x_test /= 255.
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test

def get_fashion_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    x_train /= 255.
    x_test /= 255.
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return x_train, y_train, x_test, y_test

def get_cifar10_dataset():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
        
    return  x_train, y_train, x_test, y_test
    
def train_fnn_sigmoid(file_name, dataset, layer_num, nodes_per_layer, num_epochs=5, activation = nn.sigmoid, train_with_neg_w=False):
    if dataset == 'mnist':
        x_train, y_train, x_test, y_test = get_mnist_dataset()
    elif dataset == 'fashion_mnist':
        x_train, y_train, x_test, y_test = get_fashion_mnist_dataset()
    elif dataset == 'cifar10':
        x_train, y_train, x_test, y_test = get_cifar10_dataset()
    elif dataset == 'gtsrb':
        x_train, y_train, x_test, y_test = get_GTSRB_dataset()
        
    batch_size = 128
  
    print('activation: ', activation)
    
    model = Sequential()
    
    model.add(Flatten(input_shape=x_train.shape[1:]))
    for i in range(layer_num):
        model.add(Dense(nodes_per_layer))
        
        model.add(Lambda(lambda x: nn.sigmoid(x)))
        # model.add(Lambda(lambda x: nn.tanh(x)))
        # model.add(Lambda(lambda x: tf.atan(x)))
        
    model.add(Dense(10, activation='softmax'))
    
    # sgd = SGD(lr=0.1, decay=0.1/128, momentum=0.9, nesterov=True)
    
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=sgd,
    #               metrics=['accuracy'])
    
    model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    
    print("Traing a {} layer model, saving to {}".format(layer_num + 1, file_name))
 
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              validation_data=(x_test, y_test),
              epochs=num_epochs,
              shuffle=True)
    

    # save model to a file
    if file_name != None:
        model.save(file_name+'.h5')
    
    return {'model':model, 'history':history}

def train_fnn_tanh(file_name, dataset, layer_num, nodes_per_layer, num_epochs=5, activation = nn.sigmoid, train_with_neg_w=False):
    if dataset == 'mnist':
        x_train, y_train, x_test, y_test = get_mnist_dataset()
    elif dataset == 'fashion_mnist':
        x_train, y_train, x_test, y_test = get_fashion_mnist_dataset()
    elif dataset == 'cifar10':
        x_train, y_train, x_test, y_test = get_cifar10_dataset()
    elif dataset == 'gtsrb':
        x_train, y_train, x_test, y_test = get_GTSRB_dataset()
        
    batch_size = 128
  
    print('activation: ', activation)
    
    model = Sequential()
    
    model.add(Flatten(input_shape=x_train.shape[1:]))
    for i in range(layer_num):
        model.add(Dense(nodes_per_layer))
        
        # model.add(Lambda(lambda x: nn.sigmoid(x)))
        model.add(Lambda(lambda x: nn.tanh(x)))
        # model.add(Lambda(lambda x: tf.atan(x)))
        
    model.add(Dense(10, activation='softmax'))
    
    # sgd = SGD(lr=0.1, decay=0.1/128, momentum=0.9, nesterov=True)
    
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=sgd,
    #               metrics=['accuracy'])
    
    model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    
    print("Traing a {} layer model, saving to {}".format(layer_num + 1, file_name))
 
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              validation_data=(x_test, y_test),
              epochs=num_epochs,
              shuffle=True)
    

    # save model to a file
    if file_name != None:
        model.save(file_name+'.h5')
    
    return {'model':model, 'history':history}

def train_fnn_atan(file_name, dataset, layer_num, nodes_per_layer, num_epochs=5, activation = nn.sigmoid, train_with_neg_w=False):
    if dataset == 'mnist':
        x_train, y_train, x_test, y_test = get_mnist_dataset()
    elif dataset == 'fashion_mnist':
        x_train, y_train, x_test, y_test = get_fashion_mnist_dataset()
    elif dataset == 'cifar10':
        x_train, y_train, x_test, y_test = get_cifar10_dataset()
    elif dataset == 'gtsrb':
        x_train, y_train, x_test, y_test = get_GTSRB_dataset()
        
    batch_size = 128
  
    print('activation: ', activation)
    
    model = Sequential()
    
    model.add(Flatten(input_shape=x_train.shape[1:]))
    for i in range(layer_num):
        model.add(Dense(nodes_per_layer))
        
        # model.add(Lambda(lambda x: nn.sigmoid(x)))
        # model.add(Lambda(lambda x: nn.tanh(x)))
        model.add(Lambda(lambda x: tf.atan(x)))
        
    model.add(Dense(10, activation='softmax'))
    
    # sgd = SGD(lr=0.1, decay=0.1/128, momentum=0.9, nesterov=True)
    
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=sgd,
    #               metrics=['accuracy'])
    
    model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    
    print("Traing a {} layer model, saving to {}".format(layer_num + 1, file_name))
 
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              validation_data=(x_test, y_test),
              epochs=num_epochs,
              shuffle=True)
    

    # save model to a file
    if file_name != None:
        model.save(file_name+'.h5')
    
    return {'model':model, 'history':history}
    
def train_cnn(file_name, dataset, filters, kernels, num_epochs=5, activation = nn.sigmoid, bn=False, train_with_neg_w=False):
    if dataset == 'mnist':
        x_train, y_train, x_test, y_test = get_mnist_dataset()
    elif dataset == 'fashion_mnist':
        x_train, y_train, x_test, y_test = get_fashion_mnist_dataset()
    elif dataset == 'cifar10':
        x_train, y_train, x_test, y_test = get_cifar10_dataset()
    elif dataset == 'gtsrb':
        x_train, y_train, x_test, y_test = get_GTSRB_dataset()
        
    batch_size = 128
    
    print('activation: ', activation)
    
    model = Sequential()
    model.add(Convolution2D(filters[0], kernels[0], activation=activation, input_shape=x_train.shape[1:]))
    for f, k in zip(filters[1:], kernels[1:]):
        model.add(Convolution2D(f, k, activation=activation))
        
    # the output layer, with 10 classes
    model.add(Flatten())
    if dataset == 'gtsrb':
        model.add(Dense(43, activation='softmax'))
    else:
        model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    
    print("Traing a {} layer model, saving to {}".format(len(filters) + 1, file_name))
 
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              validation_data=(x_test, y_test),
              epochs=num_epochs,
              shuffle=True)
    

    # save model to a file
    if file_name != None:
        model.save(file_name+'.h5')
    
    return {'model':model, 'history':history}
    
    
def train_lenet(file_name, dataset, params, num_epochs=10, activation=nn.sigmoid, batch_size=128, train_temp=1, pool = True):
    """
    Standard neural network training procedure. Trains LeNet-5 style model with pooling optional.
    """
    if dataset == 'mnist':
        x_train, y_train, x_test, y_test = get_mnist_dataset()
    elif dataset == 'fashion_mnist':
        x_train, y_train, x_test, y_test = get_fashion_mnist_dataset()
    elif dataset == 'cifar10':
        x_train, y_train, x_test, y_test = get_cifar10_dataset()
    elif dataset == 'gtsrb':
        x_train, y_train, x_test, y_test = get_GTSRB_dataset()
        
    img_rows, img_cols, img_channels = x_train.shape[1], x_train.shape[2], x_train.shape[3]
    input_shape = (img_rows, img_cols, img_channels)
    
    model = Sequential()
    
    model.add(Convolution2D(params[0], (5, 5), activation=activation, input_shape=input_shape, padding='same'))
    if pool:
        model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(params[1], (5, 5), activation=activation))
    if pool:
        model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(params[2], activation=activation))
    model.add(Dense(10, activation='softmax'))
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train,
              batch_size=batch_size,
              validation_data=(x_test, y_test),
              epochs=num_epochs,
              shuffle=True)
    

    if file_name != None:
        model.save(file_name+'.h5')

    return model
    
def printlog(s):
    print(s, file=open("cifar_cnn_5layer_5_3_sigmoid.txt", "a"), end='')
    
def print_weights(path_prefix, model_name):
    
    model = load_model(path_prefix + model_name, custom_objects={'fn':fn, 'tf':tf})
    model.summary()
    
    
    layer_num = 0
    
    for layer in model.layers:
        if type(layer) == Conv2D:
            printlog("layer num: {}\n".format(layer_num))
            layer_num += 1
            w,b = layer.get_weights()
            
            printlog("layer.name: {}, w.shape: {}\n".format(layer.name, w.shape))
            out_ch = w.shape[3]
            in_ch = w.shape[2]
            height = w.shape[0]
            width = w.shape[1]
            
            for i in range(out_ch):
                printlog("out_ch: {}\n".format(i))

                for j in range(in_ch):
                    printlog("in_ch: {}\n".format(j))
                    for m in range(height):
                        for n in range(width):
                            printlog("{}, ".format(w[m,n,j,i]))
                        printlog("\n")
                    printlog('---------------------------\n')
        elif (type(layer) == Dense):
            printlog("layer num: {}\n".format(layer_num))
            layer_num += 1
            w,b = layer.get_weights()
            
            printlog("layer.name: {}, w.shape: {}\n".format(layer.name, w.shape))
            out_ch = w.shape[1]
            in_ch = w.shape[0]
            
            for i in range(out_ch):
                printlog("out_ch: {}\n".format(i))
                
                for j in range(in_ch):
                    if (j%6 == 0):
                        printlog("{} \n".format(w[j,i]))
                    else:
                        printlog("{}, ".format(w[j,i]))
                    
                printlog("\n--------------------------\n")
            
            
if __name__ == '__main__':
    
    path_prefix = 'models/mixed_models/'
    
    # MNIST 数据集
    # 全连接
    # 激活函数为 sigmoid
    train_fnn_sigmoid(file_name=path_prefix+"mnist_fnn_1x100_sigmoid", dataset='mnist', 
              layer_num=1, nodes_per_layer=100, num_epochs=50)
    
    # MNIST 数据集
    # 全连接
    # 激活函数为 tanh
    train_fnn_tanh(file_name=path_prefix+"mnist_fnn_1x100_tanh", dataset='mnist', 
              layer_num=1, nodes_per_layer=100, num_epochs=50)
    
    # MNIST 数据集
    # 全连接 
    # 激活函数为 arctan
    train_fnn_atan(file_name=path_prefix+"mnist_fnn_1x100_atan", dataset='mnist', 
              layer_num=1, nodes_per_layer=100, num_epochs=50)
    
    
    # MNIST 数据集
    # 卷积
    # 激活函数为 sigmoid
    train_cnn(file_name=path_prefix+"mnist_cnn_2layer_5_3_sigmoid",dataset='mnist', 
              filters=[5], kernels = [3], num_epochs=50)
    
    # MNIST 数据集
    # 卷积
    # 激活函数为 tanh
    train_cnn(file_name=path_prefix+"mnist_cnn_2layer_5_3_tanh",dataset='mnist', 
              filters=[5], kernels = [3], num_epochs=50, activation = nn.tanh)
    
    # MNIST 数据集
    # 卷积
    # 激活函数为 arctan
    train_cnn(file_name=path_prefix+"mnist_cnn_2layer_5_3_atan",dataset='mnist', 
              filters=[5], kernels = [3], num_epochs=50, activation = tf.atan)
    
    # Fashion MNIST 数据集
    # 全连接
    # 激活函数为 sigmoid
    train_fnn_sigmoid(file_name=path_prefix+"fashion_mnist_fnn_1x100_sigmoid", dataset='fashion_mnist', 
              layer_num=1, nodes_per_layer=100, num_epochs=50)
    
    # Fashion MNIST 数据集
    # 全连接
    # 激活函数为 tanh
    train_fnn_tanh(file_name=path_prefix+"fashion_mnist_fnn_1x100_tanh", dataset='fashion_mnist', 
              layer_num=1, nodes_per_layer=100, num_epochs=50)
    
    # Fashion MNIST 数据集
    # 全连接
    # 激活函数为 arctan
    train_fnn_atan(file_name=path_prefix+"fashion_mnist_fnn_1x100_atan", dataset='fashion_mnist', 
              layer_num=1, nodes_per_layer=100, num_epochs=50)
    
    # Fashion MNIST 数据集
    # 卷积
    # 激活函数为 sigmoid
    train_cnn(file_name=path_prefix+"fashion_mnist_cnn_2layer_5_3_sigmoid", dataset='fashion_mnist', 
              filters=[5], kernels=[3], num_epochs=50)
    
    # Fashion MNIST 数据集
    # 卷积
    # 激活函数为 tanh
    train_cnn(file_name=path_prefix+"fashion_mnist_cnn_2layer_5_3_tanh",dataset='fashion_mnist', 
              filters=[5], kernels = [3], num_epochs=50, activation = nn.tanh)
    
    # Fashion MNIST 数据集
    # 卷积
    # 激活函数为 arctan
    train_cnn(file_name=path_prefix+"fashion_mnist_cnn_2layer_5_3_atan",dataset='fashion_mnist', 
              filters=[5], kernels = [3], num_epochs=50, activation = tf.atan)
    
    # Cifar10 数据集
    # 全连接
    # 激活函数为 sigmoid
    train_fnn_sigmoid(file_name=path_prefix+"cifar10_fnn_1x100_sigmoid", dataset='cifar10', 
              layer_num=1, nodes_per_layer=100, num_epochs=50)
    
    # Cifar10 数据集
    # 全连接
    # 激活函数为 tanh
    train_fnn_tanh(file_name=path_prefix+"cifar10_fnn_1x100_tanh", dataset='cifar10', 
              layer_num=1, nodes_per_layer=100, num_epochs=50)
    
    # Cifar10 数据集
    # 全连接
    # 激活函数为 arctan
    train_fnn_atan(file_name=path_prefix+"cifar10_fnn_1x100_atan", dataset='cifar10', 
              layer_num=1, nodes_per_layer=100, num_epochs=50)
    
    # Cifar10 数据集
    # 卷积
    # 激活函数为 sigmoid
    train_cnn(file_name=path_prefix+"cifar10_cnn_2layer_5_3_sigmoid", dataset='cifar10', 
              filters=[5], kernels=[3], num_epochs=50)
    
    # Cifar10 数据集
    # 卷积
    # 激活函数为 tanh
    train_cnn(file_name=path_prefix+"cifar10_cnn_2layer_5_3_tanh",dataset='cifar10', 
              filters=[5], kernels = [3], num_epochs=50, activation = nn.tanh)
    
    # Cifar10 数据集
    # 卷积
    # 激活函数为 arctan
    train_cnn(file_name=path_prefix+"cifar10_cnn_2layer_5_3_atan",dataset='cifar10', 
              filters=[5], kernels = [3], num_epochs=50, activation = tf.atan)
    
    
    
    
    
    
    
    
    
    
    
    