'''
Author: Raisa
This python script is for creating training label and save to csv format
'''

import pandas as pd
import os 
import glob 
'''data1 = {'id': [i for i in range(1,190)],  
                'label': ['Akondo']*189 }

data2 = {'id': [i for i in range(190,463)],  
                'label': ['AloeVera']*273 } 

data3 = {'id': [i for i in range(463,631)],  
                'label': ['Amloki']*168 } 

data4 = {'id': [i for i in range(631,799)],  
                'label': ['Basak']*168 } 

data5 = {'id': [i for i in range(799,925)],  
                'label': ['Kalomegh']*126} 

data6 = {'id': [i for i in range(925,1156)],  
                'label': ['Nayantara']*231 } 

data7 = {'id': [i for i in range(1156,1429)],  
                'label': ['Neem']*273 } 

data8 = {'id': [i for i in range(1429,1639)],  
                'label': ['Sojne']*210 } 

data9 = {'id': [i for i in range(1639,1891)],  
                'label': ['Thankuni']*252} 

data10 = {'id': [i for i in range(1891,2143)],  
                'label': ['Tulsi']*252 } 



df = pd.DataFrame(data1, columns = ['id','label']) 
df.to_csv('E:/CSE/Research/project/dataset/test.csv',header=True,index=False)

df = pd.DataFrame(data2, columns = ['id','label']) 
df.to_csv('E:/CSE/Research/project/dataset/test1.csv',header=True,index=False)

df = pd.DataFrame(data3, columns = ['id','label']) 
df.to_csv('E:/CSE/Research/project/dataset/test2.csv',header=True,index=False)

df = pd.DataFrame(data4, columns = ['id','label']) 
df.to_csv('E:/CSE/Research/project/dataset/test3.csv',header=True,index=False)

df = pd.DataFrame(data5, columns = ['id','label']) 
df.to_csv('E:/CSE/Research/project/dataset/test4.csv',header=True,index=False)

df = pd.DataFrame(data6, columns = ['id','label']) 
df.to_csv('E:/CSE/Research/project/dataset/test5.csv',header=True,index=False)

df = pd.DataFrame(data7, columns = ['id','label']) 
df.to_csv('E:/CSE/Research/project/dataset/test6.csv',header=True,index=False)

df = pd.DataFrame(data8, columns = ['id','label']) 
df.to_csv('E:/CSE/Research/project/dataset/test7.csv',header=True,index=False)

df = pd.DataFrame(data9, columns = ['id','label']) 
df.to_csv('E:/CSE/Research/project/dataset/test8.csv',header=True,index=False)

df = pd.DataFrame(data10, columns = ['id','label']) 
df.to_csv('E:/CSE/Research/project/dataset/test9.csv',header=True,index=False)'''

data = {'id': [i for i in range(1,2143)],  
                'label': ['Akondo']*2142 }
df = pd.DataFrame(data, columns = ['id','label']) 
df.to_csv('E:/CSE/Research/project/dataset/sample.csv',header=True,index=False)

#way to combine multiple csv file

'''os.chdir("E:/CSE/Research/project/dataset")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv( "E:/CSE/Research/project/dataset/testLabel.csv", index=False, encoding='utf-8-sig')'''