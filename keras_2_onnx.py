import torch
import torch.onnx as tonnx
from tensorflow.keras.models import load_model
import keras2onnx, onnx
import numpy as np
import torch.nn.functional as tfunc

from verinet_nn import VeriNetNN


def generate_cnn_pytorch_model(model_path, input_shape, class_num, in_channels, out_channels, kernel_sizes):
    
    keras_model_path = model_path

    # Load Keras parameters
    keras_model = load_model(keras_model_path)
    parameters = keras_model.get_weights()
    
    if len(input_shape) == 4:
        last_conv_height = input_shape[2]
        last_conv_width = input_shape[3]
    elif len(input_shape) == 3:
        last_conv_height = input_shape[1]
        last_conv_width = input_shape[2]

    # Define PyTorch layers.
    layers = []
    for inc, outc, k in zip(in_channels, out_channels, kernel_sizes):
        print("inc, outc, k: ", inc, outc, k)
        layers.append(torch.nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=k, stride=1, padding=0))
        layers.append(torch.nn.Sigmoid())
        last_conv_height = last_conv_height - k + 1
        last_conv_width = last_conv_width - k + 1
        print("last_conv_height, last_conv_width: ", last_conv_height, last_conv_width )
    
    in_feature = last_conv_height * last_conv_width * out_channels[-1]
    print("in_feature: ", in_feature)
    
    layers.append(torch.nn.Linear(in_features=in_feature, out_features=class_num))


    # Convert Keras parameters to PyTorch format.
    para_count = 0
    for i, layer in enumerate(layers):

        if isinstance(layer, torch.nn.Conv2d):
            keras_weights, keras_bias = parameters[para_count], parameters[para_count + 1]
            layers[i].weight.data = torch.Tensor(keras_weights.transpose(3, 2, 0, 1))
            layers[i].bias.data = torch.Tensor(keras_bias)
            para_count += 2

        elif isinstance(layer, torch.nn.Linear):
            keras_weights, keras_bias = parameters[para_count], parameters[para_count + 1]
            weight = torch.Tensor(keras_weights.T)

            # The following conversion should only be applied to the first dense layer after a conv layer.
            layers[i].weight.data = weight.reshape(class_num, last_conv_height, last_conv_width, out_channels[-1]).permute((0, 3, 1, 2)).reshape((class_num, in_feature))

            layers[i].bias.data = torch.Tensor(keras_bias)
            para_count += 2
            
    return layers

def generate_fnn_pytorch_model(model_path, input_shape, class_num, neurons):
    
    keras_model_path = model_path

    # Load Keras parameters
    keras_model = load_model(keras_model_path)
    parameters = keras_model.get_weights()
    
    if len(input_shape) == 4:
        input_dim = input_shape[1] * input_shape[2] * input_shape[3]
    elif len(input_shape) == 3:
        input_dim = input_shape[1] * input_shape[2]

    # Define PyTorch layers.
    layers = []
    layers.append(torch.nn.Linear(in_features=input_dim, out_features=neurons[0]))
    layers.append(torch.nn.Sigmoid())
    hidden_layer_num = len(neurons) 
    for i in range(1, hidden_layer_num):
        print("i: ", i)
        print("neurons[i] : ", neurons[i])
        layers.append(torch.nn.Linear(in_features=neurons[i-1], out_features=neurons[i]))
        layers.append(torch.nn.Sigmoid())
    
    layers.append(torch.nn.Linear(in_features=neurons[hidden_layer_num-1], out_features=class_num))


    # Convert Keras parameters to PyTorch format.
    para_count = 0
    for i, layer in enumerate(layers):

        if isinstance(layer, torch.nn.Linear):
            keras_weights, keras_bias = parameters[para_count], parameters[para_count + 1]
            weight = torch.Tensor(keras_weights.T)
            
            if para_count == 0:
                layers[i].weight.data = weight.reshape(neurons[0], 32, 32, 3).permute((0, 3, 1, 2)).reshape((neurons[0], input_dim))
            else:
                layers[i].weight.data = weight

            layers[i].bias.data = torch.Tensor(keras_bias)
            para_count += 2
            
    return layers


if __name__ == "__main__":
    
    keras_model_path = 'models/models_with_positive_weights/cifar10_ffnn_9x100_with_positive_weights_2913_cpu.h5'
    onnx_model_path = 'models/converted_for_eran/cifar10_ffnn_9x100_with_positive_weights_2913_cpu.onnx'
    
    # # transfer cnn
    # # input_shape = (1, 1, 28, 28)
    # input_shape = (1, 3, 32, 32)
    # class_num = 10
    # in_channels = [1,4]
    # out_channels = [4,4]
    # kernel_sizes = [3,3]
    
    # layers = generate_cnn_pytorch_model(keras_model_path, input_shape, class_num, in_channels, out_channels, kernel_sizes)
    
    # transfer fnn
    # input_shape = (1, 1, 28, 28)
    input_shape = (1, 3, 32, 32)
    class_num = 10
    neurons = [100, 100, 100, 100, 100, 100, 100, 100, 100]
    
    layers = generate_fnn_pytorch_model(keras_model_path, input_shape, class_num, neurons)

    # Create VeriNetNN model and save as onnx.
    verinet_model = VeriNetNN(layers)
    tonnx.export(verinet_model, torch.zeros(input_shape), onnx_model_path, verbose=False, opset_version=9)

    # check conversion
    # x = np.random.random((1, 28, 28, 1)).astype(np.float32)
    x = np.random.random((1, 32, 32, 3)).astype(np.float32)
    keras_model = load_model(keras_model_path)
    keras_pred = keras_model.predict(x).flatten()
    verinet_nn_pred = tfunc.softmax(verinet_model(torch.Tensor(x.transpose(0, 3, 1, 2))), dim=1).flatten()

    for i in range(len(keras_pred)):
        print("i: {}, keras_pred: {}, verinet_nn_pred: {}".format(i, keras_pred[i], verinet_nn_pred[i]))
        assert(abs(keras_pred[i] - verinet_nn_pred[i]) < 1e-5)

    print("Model conversion succeeded")