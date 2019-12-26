'''
Author: Raisa
Date: 26-12-19
Python Script for train data
'''
import keras
import sys
sys.path.append('../')
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator


# project modules
import config
import my_model, process_image

x_train, Y_train = process_image.load_train_data()
print("train data shape: ", x_train.shape)
print("train data label: ", Y_train.shape)

model = my_model.get_model()
model.summary()
X_train, X_test, y_train, y_test = train_test_split(x_train, Y_train, random_state=42, test_size=0.15)


#compile
opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer= opt,
           loss= keras.losses.categorical_crossentropy,
            metrics = ['accuracy'])

#checkpoints
model_cp = my_model.save_model_checkpoint()
early_stopping = my_model.set_early_stopping()

datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
                            rotation_range=10,
                            featurewise_center=True,
                            featurewise_std_normalization=True)
                            
                            
it_train = datagen.flow(x_train, y_train, batch_size=200)
steps = int(X_train.shape[0] / 200)
history = model.fit_generator(it_train, steps_per_epoch=steps, epochs=70, validation_data=(X_test, y_test), verbose=2,
            callbacks=[early_stopping,model_cp])


#history= model.fit(x_train, y_train, batch_size=200, epochs=70, verbose=2, callbacks=[early_stopping,model_cp]
 #           ,shuffle=True,validation_split=0.15)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

print("Done")
