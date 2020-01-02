'''
Author: Raisa
Date: 28-12-19
Python Script for train data
'''
import keras
import sys
sys.path.append('../')
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.layers import BatchNormalization
from keras import backend as K
from matplotlib import pyplot
#import matplotlib.pyplot as plt
# project modules
import config
import my_model, process_image
from keras.callbacks import LearningRateScheduler

# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	pyplot.savefig(config.project_root+'output/' + '_plot.png')
	pyplot.close()

x_train, Y_train = process_image.load_train_data()
print("train data shape: ", x_train.shape)
print("train data label: ", Y_train.shape)
#x_train=18627, Y_train=18627 Label


model = my_model.get_model()
model.summary()
#splitting the whole data in 0.15 size
X_train, X_test, y_train, y_test = train_test_split(x_train, Y_train, random_state=42, test_size=0.20)
#print(X_train.shape, X_test.shape,y_train.shape,y_test.shape)

#Learning rate scheduler
def lr_scheduler(epoch):
    if (epoch == 20):
        K.set_value(model.optimizer.lr, config.lr_1)
    elif (epoch == 70):
        K.set_value(model.optimizer.lr, config.lr_2)
    #elif (epoch == 60):
        #K.set_value(model.optimizer.lr, config.lr_3)
    #elif (epoch == 80):
        #K.set_value(model.optimizer.lr, config.lr_4)

    print("learning rate: ", K.get_value(model.optimizer.lr))
    return K.get_value(model.optimizer.lr)

#compile
opt = SGD(lr=0.0001, momentum=0.9)
model.compile(optimizer= opt,
           loss= keras.losses.categorical_crossentropy,
            metrics = ['accuracy'])

#checkpoints
model_cp = my_model.save_model_checkpoint()
early_stopping = my_model.set_early_stopping()

datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
                            rotation_range=5,
                            featurewise_center=True,
                            featurewise_std_normalization=True)
                            
                            
it_train = datagen.flow(X_train, y_train, batch_size=config.batch_size)
#change_lr = LearningRateScheduler(lr_scheduler)
steps = int(X_train.shape[0] / config.batch_size)
history = model.fit_generator(it_train, steps_per_epoch=steps, epochs=200, validation_data=(X_test, y_test), verbose=2,
            callbacks=[early_stopping,model_cp])


summarize_diagnostics(history)
#print ("%s: %.2f%%" % (model.metrics_names[1], history[1]*100))


#history= model.fit(x_train, y_train, batch_size=config.batch_size, epochs=70, verbose=2, callbacks=[early_stopping,model_cp]
 #           ,shuffle=True,validation_split=0.15)

print("Done")
