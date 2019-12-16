'''
Author : Raisa
Date: 24-11-19
This script is used for renaming the image files.
'''
import os
import PIL
from PIL import Image
'''os.getcwd()
collection = "E:/CSE/Research/data/dataset/"
for i, filename in enumerate(os.listdir(collection),start=1):
    os.rename("E:/CSE/Research/data/dataset/" + filename, "E:/CSE/Research/data/dataset/" + str(i) + ".jpg")
    
files='E:/CSE/Research/data/dataset'
for filename in range(1,6):
    img = Image.open('E:/CSE/Research/data/dataset/'+str(filename)+'.jpg')
#wpercent = (basewidth / float(img.size[0]))
#hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((128,128),Image.NEAREST(0))
    img.save('E:/CSE/Research/data/dataset/'+str(filename)+'.jpg')'''

images = [os.path.join('E:/CSE/Research/data/dataset/', path) for path in os.listdir('E:/CSE/Research/data/dataset/')]
for img_no, image_file in enumerate(images):
            
            img = Image.open(image_file)
            img = img.resize((128,128))
            img.save(image_file)