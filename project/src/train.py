'''
Author: Raisa
Date: 15-12-19
Python Script for train data
'''
import keras
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator

# project modules
import config
import my_model, process_image

x_train, y_train = process_image.load_train_data()
print("train data shape: ", x_train.shape)
print("train data label: ", y_train.shape)

model = my_model.get_model()
model.summary()


#compile
model.compile(optimizer= keras.optimizers.Adam(config.lr),
           loss= keras.losses.categorical_crossentropy,
            metrics = ['accuracy'])

#checkpoints
model_cp = my_model.save_model_checkpoint()
early_stopping = my_model.set_early_stopping()


model.fit(x_train, y_train, batch_size=200, epochs=50, verbose=2, callbacks=[early_stopping,model_cp]
            ,shuffle=True,validation_split=0.2)

print("Done")