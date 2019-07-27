import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

def split_data(data, ratio = [1, 0, 0], idxs=None):
    
    if idxs is None:
        num = data[0].shape[0] if type(data)==list  or type(data)==tuple else data.shape[0]
        idx = np.arange(num)
        np.random.shuffle(idx)
        idx_train = idx[:int(ratio[0]*num)]
        idx_val = idx[int(ratio[0]*num):int((ratio[0]+ratio[1])*num)]
        idx_test = idx[int((ratio[0]+ratio[1])*num):]
    else:
        idx_train,idx_val,idx_test = idxs
    train, val, test = [], [], []
    if type(data)==list or type(data)==tuple:
        for counter in range(len(data)):
            train.append(data[counter][idx_train])
            val.append(data[counter][idx_val])
            test.append(data[counter][idx_test])
    else:
        train = data[idx_train]
        val = data[idx_val]
        test = data[idx_test]
    return (train,val,test),(idx_train,idx_val,idx_test)

def get_session(number=None):
    
    config_gpu = tf.ConfigProto()
    config_gpu.gpu_options.allow_growth = True
    if number is not None:
        device = "/gpu{}".format(number)
    return tf.Session(config=config_gpu)

def read_images(files):
    ims = []
    for file in files:
        image_rgb = cv2.imread(file)
        if image_rgb.shape[0]!=image_rgb.shape[1]:
            image_rgb = image_rgb[29:-29, 9:-9,::-1]
        else:
            cut = 1
            image_rgb = cv2.resize(image_rgb[cut:-cut,cut:-cut,:], (160,160))[:,:,::-1]
        ims.append(image_rgb)
    return np.array(ims)

## From Facenet Library:
class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)
    
def get_dataset(path, has_class_directories=True):
    
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset

def get_image_paths(facedir):
    ## From Facenet Library
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths

def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat