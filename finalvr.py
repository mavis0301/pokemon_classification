from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

test_set=pd.read_csv('test_data.csv')
train_set=pd.read_csv('train_data.csv')
label=pd.read_csv('train_label.csv')

train_set = pd.get_dummies(train_set)       
test_set = pd.get_dummies(test_set)

scaler = StandardScaler()
scaler.fit(train_set)
scaled_data= scaler.transform(train_set)
scaled = pd.DataFrame(scaled_data, columns=train_set.columns[:])

traindata = scaled

scaler1 = StandardScaler()
scaler1.fit(test_set)
scaled_data1= scaler1.transform(test_set)
scaled1 = pd.DataFrame(scaled_data1, columns=test_set.columns[:])

testrow = scaled1

train=np.array(traindata)
test=np.array(testrow)


Max=[]
for i in range(len(test)):
    dis=[]
    vote=[0,0,0,0,0,0]
    for j in range(len(train)):
        dis.append(np.sum(np.square(test[i]-train[j])))
    index=np.argsort(dis)
    topk=index[:14]
    c=4
    for k in topk:
        vote[label['class'][k]]+=(c+1)
        if c>0:
            c-=1
    m=np.array(vote)
    mm=np.argsort(-m)
    Max.append(mm[0])
fileout=open('1.txt','w')
for l in Max:
    fileout.write(str(l))
    fileout.write('\n')
fileout.close()
print("END")
        
        
