'''
Author: Raisa
'''
import os
nb_train=16674
nb_test=4153
img_size=128
img_channel=1
img_shape=(img_size,img_size,img_channel)
num_classes=10
lr1=0.001
lr2=0.0001
lr3=0.00001
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
