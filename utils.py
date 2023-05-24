import numpy as np
import random
import os
import cv2
import json
from train_myself_model import *

from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import fashion_mnist

random.seed(1215)
np.random.seed(1215)

def preprocess_img(img):
    # Histogram normalization in y
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    # central scrop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    # rescale to standard size
    img = transform.resize(img, (48, 48))

    return img
    
def get_gtsrb_test_dataset():
    test = pd.read_csv('data/GT-final_test.csv',sep=';')

    x_test = []
    y_test = []
    i = 0
    for file_name, class_id  in zip(list(test['Filename']), list(test['ClassId'])):
        img_path = os.path.join('data/GTSRB/Final_Test/Images/',file_name)
        x_test.append(preprocess_img(io.imread(img_path)))
        y_test.append(class_id)
        
    x_test = np.array(x_test)
    x_test = x_test.astype('float32')
    y_test = np.array(y_test)
    y_test = to_categorical(y_test, 43)
    
    print('x_test.shape:', x_test.shape)
    print('y_test.shape:', y_test.shape)
    
    return x_test, y_test

def generate_data_myself(dataset, model, samples=10, start=0, cnn_cert_model=False, ids=None, eran_fnn=False):
    if dataset == 'gtsrb':
        x_test, y_test = get_gtsrb_test_dataset()
        print('get gtsrb test datasets')
        
    elif dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_test = x_test.astype('float32')
        x_test = np.expand_dims(x_test, axis=3)
        x_test /= 255.
        y_test = to_categorical(y_test, 10)
    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_test = x_test.astype('float32')
        x_test = np.expand_dims(x_test, axis=3)
        x_test /= 255.
        if cnn_cert_model:
            x_test -= 0.5
        if eran_fnn:
            x_test -= 0.1307
            x_test /= 0.3081
        y_test = to_categorical(y_test, 10)
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_test = x_test.astype('float32')
        x_test /= 255.
        if eran_fnn:
            x_test[:,:,:,0] -= 0.4914
            x_test[:,:,:,1] -= 0.4822
            x_test[:,:,:,2] -= 0.4465
            x_test[:,:,:,0] /= 0.2023
            x_test[:,:,:,1] /= 0.1994
            x_test[:,:,:,2] /= 0.2010
        y_test = to_categorical(y_test, 10)
    
    f = open('generate data.txt', 'a')
    inputs = []
    targets = []
    targets_labels = []
    true_labels = []
    true_ids = []
    
    print('generating labels...', file = f)
    if ids is None:
        ids = range(samples)
    else:
        ids = ids[start:start+samples]
        start = 0
    total = 0
    # traverse images
    for i in ids:
        original_predict = np.squeeze(model.predict(np.array([x_test[start+i]])))
        num_classes = len(original_predict)
        predicted_label = np.argmax(original_predict)
        print('predicted_label:', predicted_label, file = f)
        
        targets_labels = np.argsort(original_predict)[:-1]
        # sort label
        targets_labels = targets_labels[::-1]
        print('targets_labels:', targets_labels, file = f)
        
        true_label = np.argmax(y_test[start+i])
        print('true_label:', true_label, file = f)

        if true_label != predicted_label:
            continue
       
        else:
            total += 1 
            
            # images of test set
            inputs.append(x_test[start+i])
            
            true_labels.append(y_test[start+i])
            seq = []
            for c in targets_labels:
                targets.append(c)
                seq.append(c)
                
            print("[DATAGEN][L1] no = {}, true_id = {}, true_label = {}, predicted = {}, correct = {}, seq = {}".format(total, start + i, 
                np.argmax(y_test[start+i]), predicted_label, np.argmax(y_test[start+i]) == predicted_label, seq), file=f)
            
            true_ids.append(start+i)
        
    print('targets:', targets, file=f)
    #print('true_labels:', true_labels)
    # images of test set
    inputs = np.array(inputs)
    # target label
    targets = np.array(targets)
    # true label
    true_labels = np.array(true_labels)
    # id of images
    true_ids = np.array(true_ids)
    print('labels generated', file=f)
    print('{} images generated in total.'.format(len(inputs)),file=f)
    return inputs, targets, true_labels, true_ids

def process_image(dataset, image_path):
    if dataset in ['fashion_mnist', 'mnist']:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = np.expand_dims(image, axis=2)
    elif dataset in ['cifar10', 'gtsrb']:
        image = cv2.imread(image_path)
    image = image.astype('float32')
    image /= 255.
    return image

def get_dataset_from_json(dataset, num_image, json_path):
    if dataset in ['fashion_mnist', 'mnist']:
        x_test = np.zeros(shape=(num_image, 28, 28, 1))
        y_test = np.zeros(shape=(num_image, 10))
    elif dataset == 'cifar10':
        x_test = np.zeros(shape=(num_image, 32, 32, 3))
        y_test = np.zeros(shape=(num_image, 10))
        
    with open(json_path, 'r', encoding='utf8') as f:
        json_data = json.load(f)
        assert num_image == len(json_data), "the number of images in json differs with num_image"
        for i in range(num_image):
            key_str = "img_"+str(i)
            image = process_image(dataset, json_data[key_str]['path'])
            x_test[i] = image
            print('x_test.shape: ', x_test.shape)
            label = json_data[key_str]['label']
            print('label: ', label)
            if dataset in ['fashion_mnist', 'mnist', 'cifar10']:
                label = to_categorical(label, 10)
            elif dataset == 'gtsrb':
                label = to_categorical(label, 43)
            print('after label: ', label)
            y_test[i] = label
            
    return x_test, y_test

def save_original_images_from_dataset(dataset, num_images):
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    print('x_test.shape: ', x_test.shape)
        
    json_name = 'data/' + dataset + '_' + str(num_images) + '_infor.json'
    with open(json_name, 'w', encoding='utf-8') as f:
        all_infor = {}
        for i in range(num_images):
            img_dic = {}
            save_name = 'data/' + dataset + '/' + dataset + '_img_' + str(i) + '_label_' + str(y_test[i]) + '.jpg'
            img_dic['path'] = save_name
            
            cv2.imwrite(save_name, x_test[i])
                
            print('i: ', i, ', label: ', y_test[i])
            
            img_dic['label'] = int(y_test[i])
            
            img_id = "img_"+str(i)
            all_infor[img_id] = img_dic
            
        json.dump(all_infor, f, indent=2)

def generate_data(dataset, samples, data_from_local=True, targeted=True, random_and_least_likely = False, skip_wrong_label = True, start=0, 
                  cnn_cert_model=False, vnn_comp_model=False, eran_fnn=False, eran_cnn=False, ids = None, 
        target_classes = None, target_type = 0b1111, predictor = None, imagenet=False, remove_background_class=False, save_inputs=False, model_name=None, save_inputs_dir=None):
    print('data_from_local: ', data_from_local)
    
    if data_from_local:
        if dataset == 'mnist':
            json_path = 'data/mnist_'+ str(samples) +'_infor.json'
            x_test, y_test = get_dataset_from_json(dataset, samples, json_path)
            if cnn_cert_model:
                x_test -= 0.5
            if vnn_comp_model:
                x_test = x_test.reshape((x_test.shape[0], 784, 1))
                x_test = np.expand_dims(x_test, axis=1)
            if eran_fnn:
                x_test -= 0.1307
                x_test /= 0.3081
        elif dataset == 'cifar10':
            json_path = 'data/cifar10_'+ str(samples) +'_infor.json'
            x_test, y_test = get_dataset_from_json(dataset, samples, json_path)
            if cnn_cert_model:
                x_test -= 0.5
            if eran_fnn:
                x_test[:,:,:,0] -= 0.4914
                x_test[:,:,:,1] -= 0.4822
                x_test[:,:,:,2] -= 0.4465
                x_test[:,:,:,0] /= 0.2023
                x_test[:,:,:,1] /= 0.1994
                x_test[:,:,:,2] /= 0.2010
        elif dataset == 'fashion_mnist':
            json_path = 'data/fashion_mnist_'+ str(samples) +'_infor.json'
            x_test, y_test = get_dataset_from_json(dataset, samples, json_path)
        
    else:
        if dataset == 'fashion_mnist':
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
            x_test = x_test.astype('float32')
            x_test = np.expand_dims(x_test, axis=3)
            x_test /= 255.
            y_test = to_categorical(y_test, 10)
        elif dataset == 'mnist':
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_test = x_test.astype('float32')
            x_test = np.expand_dims(x_test, axis=3)
            x_test /= 255.
            if cnn_cert_model:
                x_test -= 0.5
            if vnn_comp_model:
                x_test = x_test.reshape((x_test.shape[0], 784, 1))
                x_test = np.expand_dims(x_test, axis=1)
            if eran_fnn:
                x_test -= 0.1307
                x_test /= 0.3081
            y_test = to_categorical(y_test, 10)
        elif dataset == 'cifar10':
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            x_test = x_test.astype('float32')
            x_test /= 255.
            if cnn_cert_model:
                x_test -= 0.5
            if eran_fnn:
                x_test[:,:,:,0] -= 0.4914
                x_test[:,:,:,1] -= 0.4822
                x_test[:,:,:,2] -= 0.4465
                x_test[:,:,:,0] /= 0.2023
                x_test[:,:,:,1] /= 0.1994
                x_test[:,:,:,2] /= 0.2010
            y_test = to_categorical(y_test, 10)
    
    inputs = []
    targets = []
    true_labels = []
    true_ids = []
    information = []
    target_candidate_pool = np.eye(y_test.shape[1])
    target_candidate_pool_remove_background_class = np.eye(y_test.shape[1] - 1)
    print('generating labels...')
    if ids is None:
        ids = range(samples)
    else:
        ids = ids[start:start+samples]
        if target_classes:
            target_classes = target_classes[start:start+samples]
        start = 0
    total = 0
    for i in ids:
        total += 1
        if targeted:
            predicted_label = -1 # unknown
            if random_and_least_likely:
                # if there is no user specified target classes
                if target_classes is None:
                    original_predict = np.squeeze(predictor(np.array([x_test[start+i]])))
                    num_classes = len(original_predict)
                    predicted_label = np.argmax(original_predict)
                    least_likely_label = np.argmin(original_predict)
                    top2_label = np.argsort(original_predict)[-2]
                    start_class = 1 if (imagenet and not remove_background_class) else 0
                    random_class = predicted_label
                    new_seq = [least_likely_label, top2_label, predicted_label]
                    while random_class in new_seq:
                        random_class = random.randint(start_class, start_class + num_classes - 1)
                    new_seq[2] = random_class
                    true_label = np.argmax(y_test[start+i])
                    seq = []
                    if true_label != predicted_label and skip_wrong_label:
                        seq = []
                    else:
                        if target_type & 0b10000:
                            for c in range(num_classes):
                                if c != predicted_label:
                                    seq.append(c)
                                    information.append('class'+str(c))
                        else:
                            if target_type & 0b0100:
                                # least
                                seq.append(new_seq[0])
                                information.append('least')
                            if target_type & 0b0001:
                                # top-2
                                seq.append(new_seq[1])
                                information.append('top2')
                            if target_type & 0b0010:
                                # random
                                seq.append(new_seq[2])
                                information.append('random')
                else:
                    # use user specified target classes
                    seq = target_classes[total - 1]
                    information.extend(len(seq) * ['user'])
            else:
                if imagenet:
                    if remove_background_class:
                        seq = random.sample(range(0,1000), 10)
                    else:
                        seq = random.sample(range(1,1001), 10)
                    information.extend(y_test.shape[1] * ['random'])
                else:
                    seq = range(y_test.shape[1])
                    information.extend(y_test.shape[1] * ['seq'])
            print("[DATAGEN][L1] no = {}, true_id = {}, true_label = {}, predicted = {}, correct = {}, seq = {}, info = {}".format(total, start + i, 
                np.argmax(y_test[start+i]), predicted_label, np.argmax(y_test[start+i]) == predicted_label, seq, [] if len(seq) == 0 else information[-len(seq):]))
            for j in seq:
                # skip the original image label
                if (j == np.argmax(y_test[start+i])):
                    continue
                inputs.append(x_test[start+i])
                if remove_background_class:
                    targets.append(target_candidate_pool_remove_background_class[j])
                else:
                    targets.append(target_candidate_pool[j])
                true_labels.append(y_test[start+i])
                if remove_background_class:
                    true_labels[-1] = true_labels[-1][1:]
                true_ids.append(start+i)
        else:
            true_label = np.argmax(y_test[start+i])
            original_predict = np.squeeze(predictor(np.array([x_test[start+i]])))
            num_classes = len(original_predict)
            predicted_label = np.argmax(original_predict) 
            if true_label != predicted_label and skip_wrong_label:
                continue
            else:
                inputs.append(x_test[start+i])
                if remove_background_class:
                    # shift target class by 1
                    print(np.argmax(y_test[start+i]))
                    print(np.argmax(y_test[start+i][1:1001]))
                    targets.append(y_test[start+i][1:1001])
                else:
                    targets.append(y_test[start+i])
                true_labels.append(y_test[start+i])
                if remove_background_class:
                    true_labels[-1] = true_labels[-1][1:]
                true_ids.append(start+i)
                information.extend(['original'])

    inputs = np.array(inputs)
    targets = np.array(targets)
    true_labels = np.array(true_labels)
    true_ids = np.array(true_ids)
    print('labels generated')
    print('{} images generated in total.'.format(len(inputs)))
    if save_inputs:
        if not os.path.exists(save_inputs_dir):
            os.makedirs(save_inputs_dir)
        save_model_dir = os.path.join(save_inputs_dir,model_name)
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)
        info_set = list(set(information))
        for info_type in info_set:
            save_type_dir = os.path.join(save_model_dir,info_type)
            if not os.path.exists(save_type_dir):
                os.makedirs(save_type_dir)
            counter = 0
            for i in range(len(information)):
                if information[i] == info_type:
                    df = inputs[i,:,:,0]
                    df = df.flatten()
                    np.savetxt(os.path.join(save_type_dir,'point{}.txt'.format(counter)),df,newline='\t')
                    counter += 1
            target_labels = np.array([np.argmax(targets[i]) for i in range(len(information)) if information[i]==info_type])
            np.savetxt(os.path.join(save_model_dir,model_name+'_target_'+info_type+'.txt'),target_labels,fmt='%d',delimiter='\n') 
    if eran_cnn:
        inputs -= 0.1307
        inputs /= 0.3081
    return inputs, targets, true_labels, true_ids, information

if __name__ == '__main__':
    
    # save original images of first 100 mnist dataset
    save_original_images_from_dataset(dataset = 'mnist', num_images = 100)
    
    # generate data from local
    path_prefix = "models/models_with_positive_weights/tanh/"
    file_name = path_prefix + 'mnist_ffnn_3x50_with_positive_weights_tanh_9546.h5'
    keras_model = load_model(file_name)
    inputs, targets, true_labels, true_ids, img_info = generate_data('mnist', samples=100, targeted=True, 
                                                                         random_and_least_likely = True, target_type = 0b0010, 
                                                                         predictor=keras_model.predict, start=0, 
                                                                         cnn_cert_model=False, vnn_comp_model=False, 
                                                                         eran_fnn=False, eran_cnn=False)
    
    ## save original images of first 100 cifar10 dataset
    # save_original_images_from_dataset(dataset = 'cifar10', num_images = 100)
    
    # # generate data from local
    # path_prefix = "models/models_with_positive_weights/tanh/"
    # file_name = path_prefix + 'cifar10_ffnn_3x50_with_positive_weights_tanh_38.h5'
    # keras_model = load_model(file_name)
    # inputs, targets, true_labels, true_ids, img_info = generate_data('cifar10', samples=100, targeted=True, 
    #                                                                      random_and_least_likely = True, target_type = 0b0010, 
    #                                                                      predictor=keras_model.predict, start=0, 
    #                                                                      cnn_cert_model=False, vnn_comp_model=False, 
    #                                                                      eran_fnn=False, eran_cnn=False)
    
    ## save original images of first 100 fashion_mnist dataset
    # save_original_images_from_dataset(dataset = 'fashion_mnist', num_images = 100)
    
    # # # generate data from local
    # path_prefix = "models/before_models/"
    # file_name = path_prefix + 'fashion_mnist_cnn_4layer_5_3_sigmoid_myself.h5'
    # keras_model = load_model(file_name)
    # inputs, targets, true_labels, true_ids, img_info = generate_data('fashion_mnist', samples=100, targeted=True, 
    #                                                                      random_and_least_likely = True, target_type = 0b0010, 
    #                                                                      predictor=keras_model.predict, start=0, 
    #                                                                      cnn_cert_model=False, vnn_comp_model=False, 
    #                                                                      eran_fnn=False, eran_cnn=False)

    