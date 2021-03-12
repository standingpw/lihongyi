import sys
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
data  = pd.read_csv('train.csv',encoding='big5')
#将非数值型数据转换为数值型
data[data=='NR']=0
raw_data = data.iloc[:,3:]
raw_data = raw_data.to_numpy()
#特征提取，将数据按照日期排成一行18*9*20
month_data = {}
#特征提取1
for month in range(12):
    sample = np.empty([18,480])
    for day in range(20):
        sample[:,day*24:(day+1)*24] = raw_data[18*(day+20*month):18*(day+1+20*month),:]
    month_data[month] = sample
#特征提取2
x_train_data = np.empty([12*(20*24-9),18*9],dtype=float)
y_train_data = np.empty([12*(20*24-9),1],dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day==19 and hour>14:
                continue
                #x_train_data为12*471行 18*9=162列
            x_train_data[month*471+day*24+hour,:] = month_data[month][:,day*24+hour:day*24+hour+9].reshape(1,-1)#转换为1行
                #y_train_data为12*471行，1列
            y_train_data[month*471+day*24+hour,0] = month_data[month][9,day*24+hour+9]#第9行第day*24+hour+9列
#归一化
mean_x_train_data = np.mean(x_train_data,axis=0)#按行求平均
std_x_train_data = np.mean(x_train_data,axis=0)
for i in range(len(x_train_data)):
    for j in range(len(x_train_data[0])):
        if std_x_train_data[j]!=0:
            x_train_data[i][j] = (x_train_data[i][j]-mean_x_train_data[j])/std_x_train_data[j]
#构造权值w
dim = 18*9+1
w = np.zeros([dim,1])
x_train_data = np.concatenate((np.ones([12*471,1]),x_train_data),axis=1).astype(float)
learning_rate = 100
iter_time = 3000
adagrad = np.zeros([dim,1])
eps = 0.00000001

for i in range(iter_time):
    L = np.power(np.dot(x_train_data,w)-y_train_data,2)
    loss = np.sqrt(np.sum(L)/471/12)

    if(i%100==0):
        print(str(i)+":"+str(loss))

    # gradient  = 2*np.dot(x_train_data.transpose(),np.dot(x_train_data,w)-y_train_data)
    # adagrad += gradient**2
    # w = w -learning_rate*gradient/np.sqrt(adagrad+eps)
np.save('weight.npy',w)

#预测值进行预测
test_data = pd.read_csv('test.csv',header=None,encoding='big5')
print(test_data)
test_data = test_data.iloc[:,2:]
test_data[test_data=='NR']=0
test_data = test_data.to_numpy()
test_x = np.empty([240,18*9],dtype=float)
for i in range(240):
    test_x[i,:] = test_data[18*i:18*(i+1),:].reshape(1,-1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x_train_data[j]!=0:
            test_x[i][j] = (test_x[i][j]-mean_x_train_data[j])/std_x_train_data[j]
test_x = np.concatenate((np.ones([240,1]),test_x),axis=1).astype(float)

w = np.load('weight.npy')
ans_y = np.dot(test_x,w)


import csv
with open('submit.csv',mode='w',newline='')as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id','value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row =['id_'+str(i),ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)


