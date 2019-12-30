'''
Author: Raisa
'''
import os
nb_train=18627
nb_test=2142
img_size=128
img_channel=3
img_shape=(img_size,img_size,img_channel)
num_classes=10
lr_1=0.001
lr_2=0.0001
lr_3=0.00001
#project_root='/home/cliqruser/Downloads/project/'
project_root='E:/CSE/Research/project/'

def root_path():
    #return os.path.dirname(__file__)
    return os.path.dirname(os.path.abspath(__file__))

def checkpoint_path():
    return os.path.join(root_path(),"checkpoint")

def dataset_path():
    return os.path.join(root_path(),"dataset")

def src_path():
    return os.path.join(root_path(),"src")
    
def output_path():
    return os.path.join(root_path(),"output")


#print(src_path())
