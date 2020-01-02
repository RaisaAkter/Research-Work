#Author Raisa
#Date: 30-12-19
#This file is for import image and process them so that I can use them in any deep learning algorithm
#import keras
#from keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os,sys
from PIL import Image
import cv2
from cv2 import utils
import keras
from keras.utils.np_utils import to_categorical
from keras.utils import np_utils
from keras.preprocessing import image
import numpy as np
from tqdm import tqdm
import os
import pandas as pd  
import glob
import config


def normalization(x):
    x = np.divide(x, 255.0)
    x = np.subtract(x, 0.5)
    x = np.multiply(x, 2.0)
    return x
def denoising(x): #not very useful
    img = cv2.imread(x)
    #x = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    x= cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    '''kernel = np.ones((5,5),np.uint8)
    x = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    plt.subplot(121),plt.imshow(img)'''
    plt.subplot(122),plt.imshow(x)
    plt.show()
    return x
def grayscaling(x):
    image = cv2.imread(x)
    x = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Original image',image)
    #cv2.imshow('Gray image', x)
    
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return x

def binarization(x):
    x=grayscaling(x)
    img = cv2.imread(x,0)
    #ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    #ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    #ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
    ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)# shows greater result
    #ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)#better for single leaf
    #titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
    #images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

    '''for i in range(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

    plt.show()'''
    return thresh4
def contour(x):
    im = cv2.imread(x)
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    #ret,thresh = cv2.threshold(imgray,245,255,0)
    ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_TOZERO)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    tam = 0

    for contorno in contours:
        if len(contorno) > tam:
            contornoGrande = contorno
            tam = len(contorno)
            
    x=cv2.drawContours(imgray,contornoGrande.astype('int'),-1,(0,255,0),2)

    cv2.imshow('My image',x)

    cv2.waitKey()
    cv2.destroyAllWindows()

    return x

def edgeDetection(x):
    img = cv2.imread(x,0)
    edges = cv2.Canny(img,100,200)

    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()
    return edges
#reading the label file
def load_train_data():
    train = pd.read_csv(config.project_root+'dataset/trainLabel.csv')

    #loading the train dataset
    train_image = []
    '''for i in tqdm(range(train.shape[0])):
        img = image.load_img(config.project_root+'dataset/train/'+train['id'][i].astype('str')+'.png', target_size=config.img_shape)
        #img_path=config.project_root+'dataset/train/'+train['id'][i].astype('str')+'.png'
        #imag= cv2.imread(img_path)
        #img = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
        #ret,thresh4 = cv2.threshold(imag,127,255,cv2.THRESH_TOZERO)
        img = image.img_to_array(img)  #convert to numpy array
        img = normalization(img)
        train_image.append(img) 
    X = np.array(train_image)

    np.save(config.project_root+'dataset/train'+'.npy',X)'''
    print("Loading train data...")
    X=np.load(config.project_root+'dataset/train'+'.npy')

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
    y[y == 'Thankuni'] = 8
    y[y == 'Tulsi'] = 9

    #convert the labels into categorical  

    y = np_utils.to_categorical(y,config.num_classes)
    print(X.shape[0])
    return X,y

'''for i in range(100,105):
    x=config.project_root+'dataset/train/'+str(i)+'.png'
    grayscaling(x)
    binarization(x)
    contour(x)
    edgeDetection(x)
    denoising(x)'''
#load_train_data()

