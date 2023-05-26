# DualApp
DualApp is a prototype tool for the robustness verification of neural networks. It is the official implementation for paper [A Tale of Two Approximations: Tightening Over-Approximation for DNN Robustness Verification via Under-Approximation](./A%20Tale%20of%20Two%20Approximations%20Tightening%20Over-Approximation%20for%20DNN%20Robustness%20Verification%20via%20Under-Approximation.pdf). In this project, we propose a dual-approximation approach to tighten over-approximations, leveraging an activation function's underestimated domain to define tight approximation bounds.We assess it on a comprehensive benchmark of DNNs with different architectures. Our experimental results show that DualApp significantly outperforms the state-of-the-art approaches on the verified robustness ratio and the certified lower bound. 

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


## Getting Started
***

### Start From Docker
For simplify, we provide a docker image to run:

1. Download the docker image from https://figshare.com/articles/software/DualApp/23173448.

2. Load the docker image.
    ```
    docker load -i dualapp.tar
    ```
3. Start a container with the image.
    ```
     docker run -it dualapp:v1 /bin/bash
    ```
4. Navigate to the project directory.
    ```
    cd /root/DualApp
    conda activate dualapp
    ```

5. Run demo to get parts of the results of DualApp in Table 1, including CNN4-5 on Mnist, Fashion Mnist, and FNN5*100 on Cifar-10.
    ```
    python demo.py
    ```
    After the command runs, the terminal will print out a series of related information. The result is at the last two rows of each result, like:
    ```
    [L0] method = guided_by_endpoint-sigmoid, total images = 97, avg robustness = 0.05819, avg verify runtime = 1.02, avg sample runtime = 13.86, avg GD runtime = 0.00, sample num = 1000, step = 15
    [L0] method = guided_by_endpoint-sigmoid, robustness: mean = 0.05819, std = 0.01627, var = 0.00026, max = 0.10264, min = 0.02296

    [L0] method = guided_by_endpoint-sigmoid, total images = 84, avg robustness = 0.07703, avg verify runtime = 1.06, avg sample runtime = 14.02, avg GD runtime = 0.00, sample num = 1000, step = 15
    [L0] method = guided_by_endpoint-sigmoid, robustness: mean = 0.07703, std = 0.04249, var = 0.00181, max = 0.24479, min = 0.00661

    [L0] method = guided_by_endpoint-sigmoid, total images = 52, avg robustness = 0.00370, avg verify runtime = 4.18, avg sample runtime = 13.72, avg GD runtime = 0.00, sample num = 1000, step = 15
    [L0] method = guided_by_endpoint-sigmoid, robustness: mean = 0.00370, std = 0.00166, var = 0.00000, max = 0.00773, min = 0.00098
    ```

### Install Step-By-Step

We also provide commands that will install all the necessary dependencies step by step(sudo rights might be required). 

1. Install dependencies.
    ```
    sudo apt update
    sudo apt upgrade -y
    sudo apt install build-essential zlib1g-dev libbz2-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev
    sudo apt-get install -y libgl1-mesa-dev
    sudo apt-get install libglib2.0-dev
    ```
2. Install miniconda.
    ```
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh
    ```
3. Create a virtual python environment and install all the required python dependencies(such as numpy and tensorflow).
    ```
    conda create -n dualapp python=3.7.5
    conda activate dualapp
    pip install -r requirements.txt
    pip install opencv-contrib-python==3.4.11.45
    ```
4. Modify one file of tensorflow package.
    ```
    python modify_file.py
    ```

5. Run demo to get parts of the results of DualApp in Table 1, including CNN4-5 on Mnist, Fashion Mnist, and FNN5*100 on Cifar-10.
    ```
    python demo.py
    ```


## Detailed Instructions

1. To obtain the results in Figure 8, run:
    ```
    python main_figure_8.py
    ```
    Then, record the verified robustness ratios in each experiment to a *.txt* file and run **draw_figure_8/draw.sh** to draw images. The example *.txt* files are given in **draw_figure_8/**. The first part is the perturbations, and the rests are the verified robustness ratio for DualApp on Sigmoid, DualApp on Tanh, $\alpha$-$\beta$-CROWN on Sigmoid, $\alpha$-$\beta$-CROWN on Tanh, ERAN on Sigmoid, and ERAN on Tanh, respectively.
    
    Example output of Figure 8 (models with sigmoid activation function on Mnist):
    
    | Model | Epsilon | Verified Robustness Ratio (%) |
    |:--------:|:-------------:|:-------------:|
    | FC 6*500 | 0.01 | 93.68 |
    | FC 6*500 | 0.02 | 84.21 |
    | FC 6*500 | 0.03 | 66.32 |
    | FC 6*500 | 0.04 | 46.32 |
    | FC PGD 0.1 | 0.02 | 98.00 |
    | FC PGD 0.1 | 0.04 | 97.00 |
    | FC PGD 0.1 | 0.06 | 93.00 |
    | FC PGD 0.3 | 0.04 | 92.78 |
    | FC PGD 0.3 | 0.06 | 85.57 |
    | FC PGD 0.3 | 0.08 | 78.35 |
    | FC PGD 0.3 | 0.1 | 63.92 |


2. To obtain the results in Table 1, run:
    ```
    python main_table_1.py
    ```

    Example output of Figure 8 (models with sigmoid activation function on Mnist):
    
    | Model | Certified Lower Bound |
    |:--------:|:-------------:|
    | CNN 4-5 | 0.05819 |
    | CNN 5-5 | 0.05985 |
    | CNN 6-5 | 0.06450 |
    | CNN 8-5 | 0.11412 |
    | FNN 5*100 | 0.00633 |
    | FNN 6*200 | 0.02969 |

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


The corresbonding pretrained models are provided in the folder 'pretrained_model/'. Note that we just submit models used in the experiments in body part of the paper due to the limit of supplementary material. You can refer to https://github.com/13luoyu/trained_network for other models used in Appendix. 

Results will be saved in 'logs/'. The result of FNNs will be saved in 'logs/cnn_bounds_full_with_LP_xxx.txt', and that of CNNs will be saved in 'logs/cnn_bounds_full_core_with_LP_xxx.txt'. Here xxx refers to the time stamp.



## Interface

### In cnn_bounds.py
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


### In cnn_bounds_core.py
```
run_certified_bounds_core(file_name, n_samples, p_n, q_n, data_from_local=True, method='NeWise', sample_num=0, cnn_cert_model=False, activation = 'sigmoid', mnist=False, cifar=False, fashion_mnist=False, gtsrb=False, step_tmp = 0.45, eran_fnn=False)
```
This function is used to compute the certified lower bound of a CNN.

```
run_verified_robustness_ratio_core(file_name, n_samples, p_n, q_n, data_from_local=True, method='NeWise', sample_num=0, cnn_cert_model=False, activation = 'sigmoid', mnist=False, cifar=False, fashion_mnist=False, gtsrb=False, step_tmp = 0.45, eps=0.002, eran_fnn=False)
```
This function is used to compute the verified robustness ratio of a CNN.


