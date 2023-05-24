# DualApp
DualApp is a prototype tool for the robustness verification of neural networks. It is the official implementation for paper [A Tale of Two Approximations: Tightening Over-Approximation for DNN Robustness Verification via Under-Approximation](). In this project, we propose a dual-approximation approach to tighten over-approximations, leveraging an activation function's underestimated domain to define tight approximation bounds.We assess it on a comprehensive benchmark of DNNs with different architectures. Our experimental results show that DualApp significantly outperforms the state-of-the-art approaches on the verified robustness ratio and the certified lower bound. 

## Project Structure
> - Define approximation of activation functions:
>    - activations.py
>    - cnn_bounds_core.py
>    - cnn_bounds.
>    - cnn_bounds_for_figure_3.py
> - Dataset and loader:
>    - data/
>    - utils.py
>    - setup_cifar.py
>    - setup_mnist.py
> - Benchmarks:
>    - pretrained_model/
> - Generate results in Paper:
>    - main_figure_8.py
>    - main_figure_9.py
>    - main_figure_10.py
>    - main_table_1.py
>    - main_figure_3_approximation_domain.py
>    - main_figure_3_actual_domain.py
> - Draw Pictures in Paper: 
>    - draw_figure_3/draw.py
>    - draw_figure_8/draw.py
>    - draw_figure_9/draw.py
>    - draw_figure_10/draw.py
> - Log file:
>    - logs/
> - Others


## Installation
***

### Install Step-By-Step

We first provide commands that will install all the necessary dependencies step by step(sudo rights might be required). 

1. Install dependencies:
    ```
    sudo apt update
    sudo apt upgrade -y
    sudo apt install build-essential zlib1g-dev libbz2-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev
    sudo apt-get install -y libgl1-mesa-dev
    sudo apt-get install libglib2.0-dev
    ```
2. Install miniconda
    ```
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh
    ```
3. Create a virtual python environment and install all the required python dependencies(such as numpy and tensorflow):
    ```
    conda create -n dualapp python=3.7.5
    conda activate dualapp
    pip install -r requirements.txt
    pip install opencv-contrib-python==3.4.11.45
    ```
4. Modify one file of tensorflow package:
    ```
    python modify_file.py
    ```

### Start From Docker
We provide a docker image to run:

1. Download the docker image dualapp.tar from https://figshare.com.

2. Load the docker image:
    ```
    docker load -i dualapp.tar
    ```
3. Start a container with the image:
    ```
     docker run -it dualapp:v1 /bin/bash
    ```
4. Navigate to the project directory
    ```
    cd /root/DualApp
    conda activate dualapp
    ```


## How to Run

1. To obtain the results in Figure 8, run:
    ```
    python main_figure_8.py
    ```
    Then, record the verified robustness ratios in each experiment to a *.txt* file and run **draw_figure_8/draw.py** to draw images. The example *.txt* files are given in **draw_figure_8/**. 

2. To obtain the results in Table 1, run:
    ```
    python main_table_1.py
    ```

3. To obtain the results in Figure 9, run:
    ```
    python main_figure_9.py
    ```
    Then, record the certified lower bounds and time in each experiment and write them to **draw_figure_9/draw.py** to draw images. The example datas are given in **draw_figure_9/draw.py**. 

4. To obtain the results in Figure 10, run:
    ```
    python main_figure_10.py
    ```
    Then, record the certified lower bounds and time in each experiment and write them to **draw_figure_10/draw.py** to draw images. The example datas are given in **draw_figure_10/draw.py**. 


The corresbonding pretrained models are provided in the folder 'pretrained_model/'. Note that we just submit models used in the experiments in body part of the paper due to the limit of supplementary material. You can refer to https://github.com/AnonymousAuthorsForISSTA2023/trained_network for other models used in Appendix. 

Results will be saved in 'logs/'. The result of FNNs will be saved in 'logs/cnn_bounds_full_with_LP_xxx.txt', and that of CNNs will be saved in 'logs/cnn_bounds_full_core_with_LP_xxx.txt'.


## Interface

```
run_certified_bounds(file_name, n_samples, p_n, q_n, data_from_local=True, method='NeWise', sample_num=0, cnn_cert_model=False, vnn_comp_model=False, eran_fnn=False, eran_cnn=False, activation = 'sigmoid', mnist=False, cifar=False, fashion_mnist=False, gtsrb=False, step_tmp = 0.45)
```
This function is used to compute the certified lower bound of a FNN. 
- file_name: The ".h5" file to verify. It must be a **Keras** model. 
- n_samples: Number of images to verify. 
- p_n: set it 105. 
- q_n: set it 1. 
- data_from_local: If use the images in **data/**, set it *True*, else set it *False*.
- method: The approximation approach to verify. In our method, it can be *"guided_by_endpoint"* for Monte Carlo under-approximation Algorithm or *"gradient_descent"* for gradient-based algorithm. Other methods including *"NeWise"*, *"DeepCert"*, *"VeriNet"*, and *"RobustVerifier"* are supported. 
- sample_num: If use the Monte Carlo under-approximation algorithm, it is the sample number for each image, like 1000.
- eran_fnn: As the models of Figure 8 are from ERAN, set it *True* if needed. 
- activation: The activation function used in neural network verification, like *"sigmoid"*, *"tanh"*, and *"arctan"*. 
- mnist: If the dataset is MNIST, set it *True*.
- fashion_mnist: If the dataset is Fashion MNIST, set it *True*.
- cifar: If the dataset is CIFAR-10, set it *True*.
- step_tmp: If use the gradient-based under-approximation algorithm, it is the step length of the gradient descent, like 0.45. 

```
run_verified_robustness_ratio(file_name, n_samples, p_n, q_n, data_from_local=True, method='NeWise', sample_num=0, cnn_cert_model=False, vnn_comp_model=False, eran_fnn=False, eran_cnn=False, activation = 'sigmoid', mnist=False, cifar=False, fashion_mnist=False, gtsrb=False, step_tmp = 0.45, eps=0.005)
```
This function is used to compute the verified robustness ratio of a FNN.
- file_name: The ".h5" file to verify. It must be a **Keras** model. 
- n_samples: Number of images to verify. 
- p_n: set it 105. 
- q_n: set it 1. 
- data_from_local: If use the images in **data/**, set it *True*, else set it *False*.
- method: The approximation approach to verify. In our method, it can be *"guided_by_endpoint"* for Monte Carlo under-approximation Algorithm or *"gradient_descent"* for gradient-based algorithm. Other methods including *"NeWise"*, *"DeepCert"*, *"VeriNet"*, and *"RobustVerifier"* are supported. 
- sample_num: If use the Monte Carlo under-approximation algorithm, it is the sample number for each image, like 1000.
- eran_fnn: As the models of Figure 8 are from ERAN, set it *True* if needed. 
- activation: The activation function used in neural network verification, like *"sigmoid"*, *"tanh"*, and *"arctan"*. 
- mnist: If the dataset is MNIST, set it *True*.
- fashion_mnist: If the dataset is Fashion MNIST, set it *True*.
- cifar: If the dataset is CIFAR-10, set it *True*.
- step_tmp: If use the gradient-based under-approximation algorithm, it is the step length of the gradient descent, like 0.45. 
- eps: The size of perturbation under which we verify the neural network. 


```
run_certified_bounds_core(file_name, n_samples, p_n, q_n, data_from_local=True, method='NeWise', sample_num=0, cnn_cert_model=False, activation = 'sigmoid', mnist=False, cifar=False, fashion_mnist=False, gtsrb=False, step_tmp = 0.45, eran_fnn=False)
```
This function is used to compute the certified lower bound of a CNN.

```
run_verified_robustness_ratio_core(file_name, n_samples, p_n, q_n, data_from_local=True, method='NeWise', sample_num=0, cnn_cert_model=False, activation = 'sigmoid', mnist=False, cifar=False, fashion_mnist=False, gtsrb=False, step_tmp = 0.45, eps=0.002, eran_fnn=False)
```
This function is used to compute the verified robustness ratio of a CNN.


