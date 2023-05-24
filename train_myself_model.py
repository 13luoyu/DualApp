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

from tensorflow.keras.datasets import fashion_mnist
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
    
def train_fnn(file_name, dataset, layer_num, nodes_per_layer, num_epochs=5, activation = nn.sigmoid, train_with_neg_w=False):
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
        model.add(Dense(nodes_per_layer, kernel_constraint=non_neg()))
        
        # model.add(Lambda(lambda x: nn.sigmoid(x)))
        model.add(Lambda(lambda x: nn.tanh(x)))
        
    model.add(Dense(10, kernel_constraint=non_neg(), 
                    activation='softmax'))
    
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
    model.add(Convolution2D(filters[0], kernels[0], kernel_constraint=non_neg(), activation=activation, input_shape=x_train.shape[1:]))
    for f, k in zip(filters[1:], kernels[1:]):
        model.add(Convolution2D(f, k, kernel_constraint=non_neg(), activation=activation))
    # the output layer, with 10 classes
    model.add(Flatten())
    if dataset == 'gtsrb':
        model.add(Dense(43, kernel_constraint=non_neg(), activation='softmax'))
    else:
        model.add(Dense(10, kernel_constraint=non_neg(), activation='softmax'))
    
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

class ResidualStart(Layer):
    def __init__(self, **kwargs):
        super(ResidualStart, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ResidualStart, self).build(input_shape)

    def call(self, x):
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

class ResidualStart2(Layer):
    def __init__(self, **kwargs):
        super(ResidualStart2, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ResidualStart2, self).build(input_shape)

    def call(self, x):
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

def Residual(f, activation):
    def res(x):
        x = ResidualStart()(x)
        x1 = Conv2D(f, 3, strides=1, padding='same')(x)
        x1 = BatchNormalization()(x1)
        x1 = Lambda(activation)(x1)
        x1 = Conv2D(f, 3, strides=1, padding='same')(x1)
        x1 = BatchNormalization()(x1)
        return Add()([x1, x])
    return res
   
def Residual2(f, activation):
    def res(x):
        x = ResidualStart2()(x)
        x1 = Conv2D(f, 3, strides=2, padding='same')(x)
        #x1 = BatchNormalization()(x1)
        x1 = Lambda(activation)(x1)
        x1 = Conv2D(f, 3, strides=1, padding='same')(x1)
        #x1 = BatchNormalization()(x1)
        x2 = Conv2D(f, 3, strides=2, padding='same')(x)
        #x2 = BatchNormalization()(x2)
        return Add()([x1, x2])
    return res

def train_resnet(file_name, dataset, nlayer, num_epochs=10, activation=nn.sigmoid):
    if dataset == 'mnist':
        x_train, y_train, x_test, y_test = get_mnist_dataset()
    elif dataset == 'fashion_mnist':
        x_train, y_train, x_test, y_test = get_fashion_mnist_dataset()
    elif dataset == 'cifar10':
        x_train, y_train, x_test, y_test = get_cifar10_dataset()
    elif dataset == 'gtsrb':
        x_train, y_train, x_test, y_test = get_GTSRB_dataset()
        
    print('dataset:', dataset)
    print('x_train.shape', x_train.shape)
    print('y_train.shape', y_train.shape)
        
    inputs = Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
    if nlayer == 2:
        x = Residual2(8, activation)(inputs)
        x = Lambda(activation)(x)
        x = Residual2(16, activation)(x)
        x = Lambda(activation)(x)
        x = AveragePooling2D(pool_size=7)(x)
        x = Flatten()(x)
        x = Dense(y_train.shape[1], activation='softmax')(x)
    if nlayer == 3:
        x = Residual2(8, activation)(inputs)
        x = Lambda(activation)(x)
        x = Residual(8, activation)(x)
        x = Lambda(activation)(x)
        x = Residual2(16, activation)(x)
        x = Lambda(activation)(x)
        x = AveragePooling2D(pool_size=7)(x)
        x = Flatten()(x)
        x = Dense(y_train.shape[1], activation='softmax')(x)
    if nlayer == 4:
        x = Residual2(8, activation)(inputs)
        x = Lambda(activation)(x)
        x = Residual(8, activation)(x)
        x = Lambda(activation)(x)
        x = Residual2(16, activation)(x)
        x = Lambda(activation)(x)
        x = Residual(16, activation)(x)
        x = Lambda(activation)(x)
        x = AveragePooling2D(pool_size=7)(x)
        x = Flatten()(x)
        x = Dense(y_train.shape[1], activation='softmax')(x)
    if nlayer == 5:
        x = Residual2(8, activation)(inputs)
        x = Lambda(activation)(x)
        x = Residual(8, activation)(x)
        x = Lambda(activation)(x)
        x = Residual(8, activation)(x)
        x = Lambda(activation)(x)
        x = Residual2(16, activation)(x)
        x = Lambda(activation)(x)
        x = Residual(16, activation)(x)
        x = Lambda(activation)(x)
        x = AveragePooling2D(pool_size=7)(x)
        x = Flatten()(x)
        x = Dense(y_train.shape[1], activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    # initiate the Adam optimizer
    sgd = Adam()    

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    model.summary()
  
    history = model.fit(x_train, y_train,
              batch_size=128,
              validation_data=(x_test, y_test),
              epochs=num_epochs,
              shuffle=True)
    

    # save model to a file
    if file_name != None:
        model.save(file_name+'.h5')
    
    return {'model':model, 'history':history}
    
def printlog(s):
    print(s, file=open("cifar_cnn_5layer_5_3_sigmoid.txt", "a"), end='')
    
def print_weights(path_prefix, model_name):
    
    # printlog("===============================\n")
    # printlog(path_prefix + model_name)
    # printlog("\n")
    # printlog("===============================")
    # printlog("\n")
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
    # train(file_name="models/mnist_cnn_4layer_1filter_9_11_6_sigmoid_myself",dataset='mnist', filters=[1,1,1], kernels = [9,11,6], num_epochs=10, activation = tf.nn.sigmoid)
    # train(file_name="models/mnist_cnn_5layer_5_3_sigmoid_myself",dataset='mnist', filters=[5,5,5,5], kernels = [3,3,3,3], num_epochs=10, activation = tf.nn.sigmoid)
    # train(file_name="models/mnist_cnn_6layer_5_3_sigmoid_myself",dataset='mnist', filters=[5,5,5,5,5], kernels = [3,3,3,3,3], num_epochs=10, activation = tf.nn.sigmoid)
    # train(file_name="models/mnist_cnn_10layer_5_3_sigmoid_myself",dataset='mnist', filters=[5,5,5,5,5,5,5,5,5], kernels = [3,3,3,3,3,3,3,3,3], num_epochs=10, activation = tf.nn.sigmoid)
    # train(file_name="models/fashion_mnist_cnn_4layer_5_3_sigmoid_myself",dataset='fashion_mnist', filters=[5,5,5], kernels = [3,3,3], num_epochs=10, activation = tf.nn.sigmoid)
    # train(file_name="models/fashion_mnist_cnn_5layer_5_3_sigmoid_myself",dataset='fashion_mnist', filters=[5,5,5,5], kernels = [3,3,3,3], num_epochs=10, activation = tf.nn.sigmoid)
    # train(file_name="models/fashion_mnist_cnn_6layer_5_3_sigmoid_myself",dataset='fashion_mnist', filters=[5,5,5,5,5], kernels = [3,3,3,3,3], num_epochs=10, activation = tf.nn.sigmoid)
    # train(file_name="models/fashion_mnist_cnn_8layer_5_3_sigmoid_myself",dataset='fashion_mnist', filters=[5,5,5,5,5,5,5], kernels = [3,3,3,3,3,3,3], num_epochs=10, activation = tf.nn.sigmoid)
    # train(file_name="models/fashion_mnist_cnn_10layer_5_3_sigmoid_myself",dataset='fashion_mnist', filters=[5,5,5,5,5,5,5,5,5], kernels = [3,3,3,3,3,3,3,3,3], num_epochs=10, activation = tf.nn.sigmoid)
    
    # train(file_name="models/cifar10_cnn_6layer_5_3_sigmoid_myself",dataset='cifar10', filters=[5,5,5,5,5], kernels = [3,3,3,3,3], num_epochs=10, activation = tf.nn.sigmoid)
    # train_lenet(file_name="models/cifar10_cnn_lenet_averpool_sigmoid_myself", dataset='cifar10', params=[6, 16, 100], num_epochs=10, activation = tf.nn.sigmoid)
  
    # train(file_name="models/mnist_cnn_4layer_5_3_tanh_myself",dataset='mnist', filters=[5,5,5], kernels = [3,3,3], num_epochs=10, activation = tf.nn.tanh)
    # train(file_name="models/mnist_cnn_5layer_5_3_tanh_myself",dataset='mnist', filters=[5,5,5,5], kernels = [3,3,3,3], num_epochs=10, activation = tf.nn.tanh)
    # train(file_name="models/mnist_cnn_6layer_5_3_tanh_myself",dataset='mnist', filters=[5,5,5,5,5], kernels = [3,3,3,3,3], num_epochs=10, activation = tf.nn.tanh)
    # train(file_name="models/mnist_cnn_8layer_5_3_tanh_myself",dataset='mnist', filters=[5,5,5,5,5,5,5], kernels = [3,3,3,3,3,3,3], num_epochs=10, activation = tf.nn.tanh)
    # train(file_name="models/mnist_cnn_10layer_5_3_tanh_myself",dataset='mnist', filters=[5,5,5,5,5,5,5,5,5], kernels = [3,3,3,3,3,3,3,3,3], num_epochs=10, activation = tf.nn.tanh)
    
    # train(file_name="models/fashion_mnist_cnn_4layer_5_3_tanh_myself",dataset='fashion_mnist', filters=[5,5,5], kernels = [3,3,3], num_epochs=10, activation = tf.nn.tanh)
    # train(file_name="models/fashion_mnist_cnn_5layer_5_3_tanh_myself",dataset='fashion_mnist', filters=[5,5,5,5], kernels = [3,3,3,3], num_epochs=10, activation = tf.nn.tanh)
    # train(file_name="models/fashion_mnist_cnn_6layer_5_3_tanh_myself",dataset='fashion_mnist', filters=[5,5,5,5,5], kernels = [3,3,3,3,3], num_epochs=10, activation = tf.nn.tanh)
    # train(file_name="models/fashion_mnist_cnn_8layer_5_3_tanh_myself",dataset='fashion_mnist', filters=[5,5,5,5,5,5,5], kernels = [3,3,3,3,3,3,3], num_epochs=10, activation = tf.nn.tanh)
    # train(file_name="models/fashion_mnist_cnn_10layer_5_3_tanh_myself",dataset='fashion_mnist', filters=[5,5,5,5,5,5,5,5,5], kernels = [3,3,3,3,3,3,3,3,3], num_epochs=10, activation = tf.nn.tanh)
    
    # 训练权重参数全为正数的模型
    # kernel_constraint=non_neg()
    
    # 训练全连接
    # path_prefix = "models/models_with_positive_weights/"
    
    # train_fnn(file_name=path_prefix+"mnist_ffnn_3x50_with_positive_weights", dataset='mnist', 
    #           layer_num=3, nodes_per_layer=50, num_epochs=300)
    
    # train_fnn(file_name=path_prefix+"mnist_ffnn_3x100_with_positive_weights", dataset='mnist', 
    #           layer_num=3, nodes_per_layer=100, num_epochs=200)
    
    # train_fnn(file_name=path_prefix+"mnist_ffnn_5x100_with_positive_weights", dataset='mnist', 
    #           layer_num=5, nodes_per_layer=100, num_epochs=300)
    
    # train_fnn(file_name=path_prefix+"mnist_ffnn_6x100_with_positive_weights", dataset='mnist', 
    #           layer_num=6, nodes_per_layer=100, num_epochs=300)
    
    # train_fnn(file_name=path_prefix+"mnist_ffnn_9x100_with_positive_weights", dataset='mnist', 
    #           layer_num=9, nodes_per_layer=100, num_epochs=300)
    
    # # Use Adam
    # train_fnn(file_name=path_prefix+"mnist_ffnn_3x200_with_positive_weights", dataset='mnist', 
    #           layer_num=3, nodes_per_layer=200, num_epochs=300)
    
    # # Use Adam
    # train_fnn(file_name=path_prefix+"mnist_ffnn_3x60_with_positive_weights", dataset='mnist', 
    #           layer_num=3, nodes_per_layer=60, num_epochs=300)
    
    # 训练卷积
    
    # train_cnn(file_name=path_prefix+"mnist_cnn_3layer_2_3_with_positive_weights",dataset='mnist', 
    #           filters=[2,2], kernels = [3,3], num_epochs=50)
    
    # train_cnn(file_name=path_prefix+"mnist_cnn_3layer_4_3_with_positive_weights",dataset='mnist', 
    #           filters=[4,4], kernels = [3,3], num_epochs=50)
    
    # train_cnn(file_name=path_prefix+"mnist_cnn_4layer_5_3_with_positive_weights",dataset='mnist', 
    #           filters=[5,5,5], kernels = [3,3,3], num_epochs=50)
    
    # train_cnn(file_name=path_prefix+"mnist_cnn_6layer_5_3_with_positive_weights",dataset='mnist', 
    #           filters=[5,5,5,5,5], kernels = [3,3,3,3,3], num_epochs=50)
    
    # train_cnn(file_name=path_prefix+"mnist_cnn_8layer_5_3_with_positive_weights",dataset='mnist', 
    #           filters=[5,5,5,5,5,5,5], kernels = [3,3,3,3,3,3,3], num_epochs=50)
    
    # train_cnn(file_name=path_prefix+"mnist_cnn_10layer_5_3_with_positive_weights",dataset='mnist', 
    #           filters=[5,5,5,5,5,5,5,5,5], kernels = [3,3,3,3,3,3,3,3,3], num_epochs=50)
    
    # train_cnn(file_name=path_prefix+"mnist_cnn_8layer_35_3_with_positive_weights",dataset='mnist', 
    #           filters=[35,35,35,35,35,35,35], kernels = [3,3,3,3,3,3,3], num_epochs=50)
    
    # 训练权重参数全为负数的模型
    # kernel_constraint=max_norm(0.0)
    
    # # 训练全连接
    # path_prefix = "models/models_with_negative_weights/"
    
    # train_fnn(file_name=path_prefix+"mnist_ffnn_2x50_with_negative_weights", dataset='mnist', 
    #           layer_num=2, nodes_per_layer=50, num_epochs=300, train_with_neg_w=True)
    
    # 训练卷积
    
    # train_cnn(file_name=path_prefix+"mnist_cnn_3layer_2_3_with_positive_weights",dataset='mnist', 
    #           filters=[2,2], kernels = [3,3], num_epochs=50, train_with_neg_w=True)
    
    # 找正负参杂模型的规律
    # path_prefix = "models/public_benchmarks/mnist/"
    # model_name = "mnist_cnn_8layer_5_3_sigmoid"
    # print_weights(path_prefix, model_name)
    
    # path_prefix = "models/public_benchmarks/mnist/"
    # model_name = "mnist_cnn_7layer_sigmoid"
    # print_weights(path_prefix, model_name)
    
    # path_prefix = "models/before_models/"
    # model_name = "fashion_mnist_cnn_4layer_5_3_sigmoid_myself.h5"
    # print_weights(path_prefix, model_name)
    
    # 训练权重参数全为正数的模型 - tanh

    # 训练全连接 mnist
    path_prefix = "models/models_with_positive_weights/tanh/"
    
    # train_fnn(file_name=path_prefix+"mnist_ffnn_3x50_with_positive_weights_tanh", dataset='mnist', 
    #           layer_num=3, nodes_per_layer=50, num_epochs=300, activation = nn.tanh)
    
    # train_fnn(file_name=path_prefix+"mnist_ffnn_3x100_with_positive_weights_tanh", dataset='mnist', 
    #           layer_num=3, nodes_per_layer=100, num_epochs=200)
    
    # train_fnn(file_name=path_prefix+"mnist_ffnn_3x200_with_positive_weights_tanh", dataset='mnist', 
    #           layer_num=3, nodes_per_layer=200, num_epochs=200)
    
    # train_fnn(file_name=path_prefix+"mnist_ffnn_5x100_with_positive_weights_tanh", dataset='mnist', 
    #           layer_num=5, nodes_per_layer=100, num_epochs=300)
    
    # 训练全连接 cifar10
    # path_prefix = "models/models_with_positive_weights/tanh/"
    
    # train_fnn(file_name=path_prefix+"cifar10_ffnn_3x50_with_positive_weights_tanh", dataset='cifar10', 
    #           layer_num=3, nodes_per_layer=50, num_epochs=300, activation = nn.tanh)
    
    # train_fnn(file_name=path_prefix+"cifar10_ffnn_3x100_with_positive_weights_tanh", dataset='cifar10', 
    #           layer_num=3, nodes_per_layer=100, num_epochs=300)
    
    # train_fnn(file_name=path_prefix+"cifar10_ffnn_3x200_with_positive_weights_tanh", dataset='cifar10', 
    #           layer_num=3, nodes_per_layer=200, num_epochs=300)
    
    # train_fnn(file_name=path_prefix+"cifar10_ffnn_5x100_with_positive_weights_tanh", dataset='cifar10', 
    #           layer_num=5, nodes_per_layer=100, num_epochs=300)
    
    # 训练卷积 mnist
    
    # train_cnn(file_name=path_prefix+"mnist_cnn_3layer_2_3_with_positive_weights_tanh",dataset='mnist', 
    #           filters=[2,2], kernels = [3,3], num_epochs=100, activation = nn.tanh)
    
    # train_cnn(file_name=path_prefix+"mnist_cnn_3layer_4_3_with_positive_weights_tanh",dataset='mnist', 
    #           filters=[4,4], kernels = [3,3], num_epochs=50, activation = nn.tanh)
    
    # train_cnn(file_name=path_prefix+"mnist_cnn_4layer_5_3_with_positive_weights_tanh",dataset='mnist', 
    #           filters=[5,5,5], kernels = [3,3,3], num_epochs=50, activation = nn.tanh)
    
    # train_cnn(file_name=path_prefix+"mnist_cnn_5layer_5_3_with_positive_weights_tanh",dataset='mnist', 
    #           filters=[5,5,5,5], kernels = [3,3,3,3], num_epochs=50, activation = nn.tanh)
    
    # train_cnn(file_name=path_prefix+"mnist_cnn_6layer_5_3_with_positive_weights_tanh",dataset='mnist', 
    #           filters=[5,5,5,5,5], kernels = [3,3,3,3,3], num_epochs=50, activation = nn.tanh)
    
    # 训练卷积 cifar10
    
    train_cnn(file_name=path_prefix+"cifar10_cnn_3layer_2_3_with_positive_weights_tanh",dataset='cifar10', 
              filters=[2,2], kernels = [3,3], num_epochs=100, activation = nn.tanh)
    
    # train_cnn(file_name=path_prefix+"cifar10_cnn_3layer_4_3_with_positive_weights_tanh",dataset='cifar10', 
    #           filters=[4,4], kernels = [3,3], num_epochs=50, activation = nn.tanh)
    
    # train_cnn(file_name=path_prefix+"cifar10_cnn_3layer_5_3_with_positive_weights_tanh",dataset='cifar10', 
    #           filters=[5,5], kernels = [3,3], num_epochs=50, activation = nn.tanh)
    
    # 训练权重参数全为正数的模型 - atan

    # 训练全连接 mnist
    # path_prefix = "models/models_with_positive_weights/arctan/"
    
    # train_fnn(file_name=path_prefix+"mnist_ffnn_3x50_with_positive_weights_atan", dataset='mnist', 
    #           layer_num=3, nodes_per_layer=50, num_epochs=100, activation = tf.atan)
    
    # train_fnn(file_name=path_prefix+"mnist_ffnn_3x100_with_positive_weights_atan", dataset='mnist', 
    #           layer_num=3, nodes_per_layer=100, num_epochs=200, activation = tf.atan)
    
    # train_fnn(file_name=path_prefix+"mnist_ffnn_3x200_with_positive_weights_atan", dataset='mnist', 
    #           layer_num=3, nodes_per_layer=200, num_epochs=200, activation = tf.atan)
    
    # train_fnn(file_name=path_prefix+"mnist_ffnn_5x100_with_positive_weights_atan", dataset='mnist', 
    #           layer_num=5, nodes_per_layer=100, num_epochs=200, activation = tf.atan)
    
    # train_fnn(file_name=path_prefix+"mnist_ffnn_3x400_with_positive_weights_atan", dataset='mnist', 
    #           layer_num=3, nodes_per_layer=400, num_epochs=200, activation = tf.atan)
    
    # train_fnn(file_name=path_prefix+"mnist_ffnn_3x700_with_positive_weights_atan", dataset='mnist', 
    #           layer_num=3, nodes_per_layer=700, num_epochs=200, activation = tf.atan)
    
    # 训练卷积 cifar10
    
    # train_fnn(file_name=path_prefix+"cifar10_ffnn_3x50_with_positive_weights_atan", dataset='cifar10', 
    #           layer_num=3, nodes_per_layer=50, num_epochs=300, activation = tf.atan)
    
    # train_fnn(file_name=path_prefix+"cifar10_ffnn_3x100_with_positive_weights_atan", dataset='cifar10', 
    #           layer_num=3, nodes_per_layer=100, num_epochs=300, activation = tf.atan)
    
    # train_fnn(file_name=path_prefix+"cifar10_ffnn_3x200_with_positive_weights_atan", dataset='cifar10', 
    #           layer_num=3, nodes_per_layer=200, num_epochs=300, activation = tf.atan)
    
    # train_fnn(file_name=path_prefix+"cifar10_ffnn_4x100_with_positive_weights_atan", dataset='cifar10', 
    #           layer_num=4, nodes_per_layer=100, num_epochs=300, activation = tf.atan)
    
    # train_fnn(file_name=path_prefix+"cifar10_ffnn_5x100_with_positive_weights_atan", dataset='cifar10', 
    #           layer_num=5, nodes_per_layer=100, num_epochs=300, activation = tf.atan)
    
    # train_fnn(file_name=path_prefix+"cifar10_ffnn_6x100_with_positive_weights_atan", dataset='cifar10', 
    #           layer_num=6, nodes_per_layer=100, num_epochs=300, activation = tf.atan)
    
    # train_fnn(file_name=path_prefix+"cifar10_ffnn_9x100_with_positive_weights_atan", dataset='cifar10', 
    #           layer_num=9, nodes_per_layer=100, num_epochs=800, activation = tf.atan)
    
    # train_fnn(file_name=path_prefix+"cifar10_ffnn_3x400_with_positive_weights_atan", dataset='cifar10', 
    #           layer_num=3, nodes_per_layer=400, num_epochs=300, activation = tf.atan)
    
    # train_fnn(file_name=path_prefix+"cifar10_ffnn_3x700_with_positive_weights_atan", dataset='cifar10', 
    #           layer_num=3, nodes_per_layer=700, num_epochs=300, activation = tf.atan)
    
    
    
    # 训练卷积 mnist
    
    # train_cnn(file_name=path_prefix+"mnist_cnn_3layer_2_3_with_positive_weights_atan",dataset='mnist', 
    #           filters=[2,2], kernels = [3,3], num_epochs=100, activation = tf.atan)
    
    # train_cnn(file_name=path_prefix+"mnist_cnn_3layer_4_3_with_positive_weights_atan",dataset='mnist', 
    #           filters=[4,4], kernels = [3,3], num_epochs=50, activation = tf.atan)
    
    # train_cnn(file_name=path_prefix+"mnist_cnn_4layer_5_3_with_positive_weights_atan",dataset='mnist', 
    #           filters=[5,5,5], kernels = [3,3,3], num_epochs=50, activation = tf.atan)
    
    # train_cnn(file_name=path_prefix+"mnist_cnn_5layer_5_3_with_positive_weights_atan",dataset='mnist', 
    #           filters=[5,5,5,5], kernels = [3,3,3,3], num_epochs=50, activation = tf.atan)
    
    # train_cnn(file_name=path_prefix+"mnist_cnn_6layer_5_3_with_positive_weights_atan",dataset='mnist', 
    #           filters=[5,5,5,5,5], kernels = [3,3,3,3,3], num_epochs=50, activation = tf.atan)
    
    # 训练卷积 cifar10
    
    # train_cnn(file_name=path_prefix+"cifar10_cnn_3layer_2_3_with_positive_weights_atan",dataset='cifar10', 
    #           filters=[2,2], kernels = [3,3], num_epochs=100, activation = tf.atan)
    
    # train_cnn(file_name=path_prefix+"cifar10_cnn_3layer_4_3_with_positive_weights_atan",dataset='cifar10', 
    #           filters=[4,4], kernels = [3,3], num_epochs=100, activation = tf.atan)
    
    # train_cnn(file_name=path_prefix+"cifar10_cnn_3layer_5_3_with_positive_weights_atan",dataset='cifar10', 
    #           filters=[5,5], kernels = [3,3], num_epochs=50, activation = tf.atan)
    