'''
Author: Raisa
This python script is for creating training label and save to csv format
'''

import pandas as pd
import os 
import glob 
'''data1 = {'id': [i for i in range(1,1555)],  
                'label': ['Akondo']*1554 }
data2 = {'id': [i for i in range(1555,3634)],  
                'label': ['Aloe Vera']*2079 } 

data3 = {'id': [i for i in range(3634,4936)],  
                'label': ['Amloki']*1302 } 

data4 = {'id': [i for i in range(4936,6301)],  
                'label': ['Basak']*1365 } 
data5 = {'id': [i for i in range(6301,7309)],  
                'label': ['Kalomegh']*1008 } 
data6 = {'id': [i for i in range(7309,9115)],  
                'label': ['Nayantara']*1806 } 
data7 = {'id': [i for i in range(9115,11194)],  
                'label': ['Neem']*2079 } 
data8 = {'id': [i for i in range(11194,12811)],  
                'label': ['Sojne']*1617 } 
data9 = {'id': [i for i in range(12811,14764)],  
                'label': ['thankuni']*1953} 
data10 = {'id': [i for i in range(14764,16675)],  
                'label': ['Tulsi']*1911 } 



df = pd.DataFrame(data1, columns = ['id','label']) 
df.to_csv('E:/CSE/Research/project/dataset/train.csv',header=True,index=False)

df = pd.DataFrame(data2, columns = ['id','label']) 
df.to_csv('E:/CSE/Research/project/dataset/train1.csv',header=True,index=False)

df = pd.DataFrame(data3, columns = ['id','label']) 
df.to_csv('E:/CSE/Research/project/dataset/train2.csv',header=True,index=False)

df = pd.DataFrame(data4, columns = ['id','label']) 
df.to_csv('E:/CSE/Research/project/dataset/train3.csv',header=True,index=False)

df = pd.DataFrame(data5, columns = ['id','label']) 
df.to_csv('E:/CSE/Research/project/dataset/train4.csv',header=True,index=False)

df = pd.DataFrame(data6, columns = ['id','label']) 
df.to_csv('E:/CSE/Research/project/dataset/train5.csv',header=True,index=False)

df = pd.DataFrame(data7, columns = ['id','label']) 
df.to_csv('E:/CSE/Research/project/dataset/train6.csv',header=True,index=False)

df = pd.DataFrame(data8, columns = ['id','label']) 
df.to_csv('E:/CSE/Research/project/dataset/train7.csv',header=True,index=False)

df = pd.DataFrame(data9, columns = ['id','label']) 
df.to_csv('E:/CSE/Research/project/dataset/train8.csv',header=True,index=False)

df = pd.DataFrame(data10, columns = ['id','label']) 
df.to_csv('E:/CSE/Research/project/dataset/train9.csv',header=True,index=False)'''

#way to combine multiple csv file

os.chdir("E:/CSE/Research/project/dataset")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv( "E:/CSE/Research/project/dataset/trainLabels.csv", index=False, encoding='utf-8-sig')