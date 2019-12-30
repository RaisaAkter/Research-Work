'''
Date:29-12-19
Author:Raisa
This python script is used for loading a saved model and try to predict data with that model
'''
from sklearn.model_selection import train_test_split
import config
import process_image
import keras
from keras.models import load_model
from keras.optimizers import SGD
import pandas as pd
import test 

test = test.load_test_data()
#X_train, X_test, y_train, y_test = train_test_split(x_train, Y_train, random_state=42, test_size=0.15)

model = load_model(config.project_root+'checkpoint/'+'baseline0.h5')
#loaded_model.load_weights("Data/model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data 
# Define X_test & Y_test data first

'''opt=SGD(lr=0.001, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
score = model.evaluate(X_test, y_test, verbose=0)#data have to check
print ("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))'''

def replace_error(data):
    data.replace(1,'AloeVera', inplace=True)
    data.replace(2,'Amloki', inplace=True)
    data.replace(3,'Basak', inplace=True)
    data.replace(4,'Kalomegh', inplace=True)
    data.replace(5,'Nayantara', inplace=True)
    data.replace(6,'Neem', inplace=True)
    data.replace(7,'Sojne', inplace=True)
    data.replace(8,'Thankuni', inplace=True)
    data.replace(9,'Tulsi', inplace=True)
    data.replace(0,'Akondo', inplace=True)

# making predictions
prediction = model.predict_classes(test, batch_size=200, verbose=2)
# creating submission file
sample = pd.read_csv(config.project_root+'dataset/sample.csv')
sample['label'] = prediction
sample.to_csv(config.project_root+'output/prediction'+'.csv', header=True, index=False)
data=pd.read_csv(config.project_root+'output/prediction'+'.csv')
replace_error(data)
data.to_csv(config.project_root+'output/final_prediction'+'.csv',header=True,index=False)
print("Complete")