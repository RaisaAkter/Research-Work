'''
Date:30-12-19
Author:Raisa
This script is used for loading test images
'''
import config,process_image
import keras
import numpy as np 
from keras.utils import np_utils
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tqdm import tqdm
import os

def load_test_data():

    test_image = []
    '''for i in tqdm(range(1,config.nb_test+1)):
        img = image.load_img(config.project_root+'dataset/test/'+str(i)+'.png', target_size=config.img_shape)
        img = image.img_to_array(img)
        img = process_image.normalization(img)
        test_image.append(img)
    y = np.array(test_image)
    np.save(config.project_root+'dataset/test'+'.npy',y)'''
    print("Loading test data...")
    y=np.load(config.project_root+'dataset/test'+'.npy')
    return y 
