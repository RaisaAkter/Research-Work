'''
Author : Raisa
Date: 24-11-19
This script is used for renaming the image files.
'''
import os
import PIL
from PIL import Image


os.getcwd()
collection = "E:/CSE/Thesis/project/dataset/augmented_images/augment_tulsi/"
for i, filename in enumerate(os.listdir(collection),start=14764):
    os.rename("E:/CSE/Thesis/project/dataset/augmented_images/augment_tulsi/" 
            + filename, "E:/CSE/Thesis/project/dataset/augmented_images/augment_tulsi/" + str(i) + ".png")
    