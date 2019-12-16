#Author Raisa
#Date: 15-12-19
#This file is for import image and process them so that I can use them in any deep learning algorithm
#import keras
#from keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os,sys
from PIL import Image
import cv2
import keras
from keras.utils.np_utils import to_categorical
from keras.utils import np_utils
from keras.preprocessing import image
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import pandas as pd  
import glob

#from .. import config

#Basic information
num_classes=10
nb_train_img=16674
nb_test_img=4153

def normalization(x):
    x = np.divide(x, 255.0)
    x = np.subtract(x, 0.5)
    x = np.multiply(x, 2.0)
    return x

#reading the label file
def load_train_data():
    train = pd.read_csv('E:/CSE/Thesis/project/dataset/trainLabel.csv')

    #loading the train dataset
    train_image = []
    train_image1 = []
    train_image2 = []
    '''for i in tqdm(range(train.shape[0]//3)):
        img = image.load_img('E:/CSE/Thesis/project/dataset/train/'+train['id'][i].astype('str')+'.png', target_size=(128,128,3))
        img = image.img_to_array(img)  #convert to numpy array
        img = normalization(img)
        train_image.append(img) 
    X1 = np.array(train_image)
    np.save('E:/CSE/Thesis/project/dataset/train1'+'.npy',X1)

    for i in tqdm(range(train.shape[0]//3,(train.shape[0]*2)//3)):
        img = image.load_img('E:/CSE/Thesis/project/dataset/train/'+train['id'][i].astype('str')+'.png', target_size=(128,128,3))
        img = image.img_to_array(img)  #convert to numpy array
        img = normalization(img)
        train_image1.append(img) 
    X2 = np.array(train_image1)
    np.save('E:/CSE/Thesis/project/dataset/train2'+'.npy',X2)

    for i in tqdm(range((train.shape[0]*2)//3,train.shape[0])):
        img = image.load_img('E:/CSE/Thesis/project/dataset/train/'+train['id'][i].astype('str')+'.png', target_size=(128,128,3))
        img = image.img_to_array(img)  #convert to numpy array
        img = normalization(img)
        train_image2.append(img) 
    X3 = np.array(train_image2)
    np.save('E:/CSE/Thesis/project/dataset/train3'+'.npy',X3)'''

    '''fpath ='E:/CSE/Thesis/project/dataset/train'
    npyfilespath = 'E:/CSE/Thesis/project/dataset/' 
    os.chdir(npyfilespath)
    npfiles= glob.glob("*.npy")
    npfiles.sort()
    all_arrays = []
    for i, npfile in enumerate(npfiles):
        all_arrays.append(np.load(os.path.join(npyfilespath, npfile)))
    np.save(fpath+'.npy', np.concatenate(all_arrays))
    print("Loading data...")
    train1=np.load('E:/CSE/Thesis/project/dataset/train1'+'.npy')
    train2=np.load('E:/CSE/Thesis/project/dataset/train2'+'.npy')
    train3=np.load('E:/CSE/Thesis/project/dataset/train3'+'.npy')
    concat=np.concatenate((train1,train2,train3),axis=0)
    np.save('E:/CSE/Thesis/project/dataset/train'+'.npy',concat)

    #np.save('E:/CSE/Thesis/project/dataset/train'+'.npy',X)'''
    print("Loading train data...")
    X=np.load('E:/CSE/Thesis/project/dataset/train1'+'.npy')

    #on hot encoding
    y=train['label'].values
    y[y == 'Akondo'] = 0
    y[y == 'AloeVera'] = 1
    y[y == 'Amloki'] = 2
    y[y == 'Basak'] = 3
    y[y == 'Kalomegh'] = 4
    y[y == 'Nayantara'] = 5
    y[y == 'Neem'] = 6
    y[y == 'Sojne'] = 7
    y[y == 'thankuni'] = 8
    y[y == 'Tulsi'] = 9

    #convert the labels into categorical  

    y = np_utils.to_categorical(y,num_classes)
    return X,y
#load_train_data()
#Split the data
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
#print(X.shape[0])
#print('Complete')