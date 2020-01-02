'''
Date:31-12-19
Author: Raisa
This script is used for calculating the test accuracy
'''
import pandas as pd
import config 

c=0
test=pd.read_csv(config.project_root+'output/final_prediction'+'.csv')
original=pd.read_csv(config.project_root+'dataset/testLabel'+'.csv')
for i in range(0,2142):
    if original['label'][i]!=test['label'][i]:
        c=c+1
accuracy=100-(c/2142)*100
print('Test Accuracy: %.2f%%'% accuracy)
print(c) #number of wrong prediction