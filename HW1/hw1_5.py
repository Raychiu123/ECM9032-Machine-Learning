
# coding: utf-8

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv
from math import exp
import scipy.io as sio
import numpy as np
target = sio.loadmat('C:\\Users\\gaexp\\OneDrive\\Documents\\Machine Learning\\Homework\\[2017]ML_HW1\\data\\5_T.mat')
X = sio.loadmat('C:\\Users\\gaexp\\OneDrive\\Documents\\Machine Learning\\Homework\\[2017]ML_HW1\\data\\5_X.mat')


# In[2]:

def rmse(predictions, targets): #Root Mean Square
    return np.sqrt(((np.array(predictions) - np.array(targets)) ** 2).mean())


# In[3]:

train_target1 = target['T'][0:40]
train_target2 = target['T'][50:90]
train_target3 = target['T'][100:140]
train_target = np.append(train_target1,train_target2)
train_target = np.append(train_target,train_target3) #train_target
test_target1 = target['T'][40:50]
test_target2 = target['T'][90:100]
test_target3 = target['T'][140:150]
test_target = np.append(test_target1,test_target2)
test_target = np.append(test_target, test_target3) #test_target

train_X1 = X['X'][0:40]
train_X2 = X['X'][50:90]
train_X3 = X['X'][100:140]
train_x = np.append(train_X1,train_X2, axis=0)
train_x = np.append(train_x,train_X3, axis=0) #train_X

test_X1 = X['X'][40:50]
test_X2 = X['X'][90:100]
test_X3 = X['X'][140:150]
test_x = np.append(test_X1,test_X2, axis=0)
test_x = np.append(test_x,test_X3, axis=0) #test_X


# In[4]:

func=[]
func1=[]
func2=[]
func3=[]
func4=[]
func5=[]
func6=[]
func7=[]
func8=[]
func9=[]
for i in range(120): 
    func.append(1)
    func1.append(train_x[i][0])
    func2.append(train_x[i][1])
    func3.append(train_x[i][2])
    func4.append(train_x[i][3])
fi1 = np.array([func,func1,func2,func3,func4]) #Basis function for M=1(train)
for i in range(30): 
    func5.append(1)
    func6.append(test_x[i][0])
    func7.append(test_x[i][1])
    func8.append(test_x[i][2])
    func9.append(test_x[i][3])
fi2 = np.array([func5,func6,func7,func8,func9]) #Basis function for M=1(test)


# In[5]:

w1 = np.dot(np.dot(inv(np.dot(fi1,fi1.transpose())),fi1),train_target) #w_ml for M=1
w0_1 = np.array(train_x).mean()-np.dot(w1[1:],np.array([fi1[1].mean(),fi1[2].mean(),fi1[3].mean(),fi1[4].mean()])).mean() #w0_ml for M=1


# In[6]:

train_rmse1 = rmse(w0_1+np.dot(w1[1:],fi1[1:]),train_target)
test_rmse1 = rmse(w0_1+np.dot(w1[1:],fi2[1:]),test_target)
print(test_rmse1,train_rmse1)


# In[7]:

func10=[]
func11=[]
func12=[]
func13=[]
func14=[]
func15=[]
func16=[]
func17=[]
func18=[]
func19=[]
for i in range(120):
    func10.append(train_x[i][0]**2)
    func11.append(train_x[i][0]*train_x[i][1])
    func12.append(train_x[i][0]*train_x[i][2])
    func13.append(train_x[i][0]*train_x[i][3])
    func14.append(train_x[i][1]*train_x[i][1])
    func15.append(train_x[i][1]*train_x[i][2])
    func16.append(train_x[i][1]*train_x[i][3])
    func17.append(train_x[i][2]*train_x[i][2])
    func18.append(train_x[i][2]*train_x[i][3])
    func19.append(train_x[i][3]*train_x[i][3])
    
fi3 = np.array([func,func1,func2,func3,func4,func10,func11,func12,func13,func11,func14,func15,func16,func12,func15,func17,func18,func13,func16,func18,func19])
fi3 = fi3.transpose()  #Basis function for M=2(train)

func20=[]
func21=[]
func22=[]
func23=[]
func24=[]
func25=[]
func26=[]
func27=[]
func28=[]
func29=[]
for i in range(30):
    func20.append(test_x[i][0]**2)
    func21.append(test_x[i][0]*test_x[i][1])
    func22.append(test_x[i][0]*test_x[i][2])
    func23.append(test_x[i][0]*test_x[i][3])
    func24.append(test_x[i][1]*test_x[i][1])
    func25.append(test_x[i][1]*test_x[i][2])
    func26.append(test_x[i][1]*test_x[i][3])
    func27.append(test_x[i][2]*test_x[i][2])
    func28.append(test_x[i][2]*test_x[i][3])
    func29.append(test_x[i][3]*test_x[i][3])
    
fi4 = np.array([func5,func6,func7,func8,func9,func20,func21,func22,func23,func21,func24,func25,func26,func22,func25,func27,func28,func23,func26,func28,func29])
fi4 = fi4.transpose() #Basis function for M=2(test)


# In[8]:

w2 = np.dot(np.dot(inv(np.dot(fi3.transpose(),fi3)),fi3.transpose()),train_target)
w0_2 = (train_x).mean()-np.dot(w2[1:],np.array([fi3[1].mean(),fi3[2].mean(),fi3[3].mean(),fi3[4].mean(),fi3[5].mean(),fi3[6].mean(),fi3[7].mean(),fi3[8].mean(),fi3[9].mean(),fi3[10].mean(),fi3[11].mean(),fi3[12].mean(),fi3[13].mean(),fi3[14].mean(),fi3[15].mean(),fi3[16].mean(),fi3[17].mean(),fi3[18].mean(),fi3[19].mean(),fi3[20].mean()]))


# In[9]:

train_rmse2 = rmse(w0_2+np.dot(w2[1:],fi3.transpose()[1:]),train_target)
test_rmse2 = rmse(w0_2+np.dot(w2[1:],fi4.transpose()[1:]),test_target)
print(test_rmse2,train_rmse2)


# In[10]:

fi5 = np.array([func,func1,func10]) #for attribute1
fi5 = fi5.transpose()
w3_0 = np.dot(np.dot(inv(np.dot(fi5.transpose(),fi5)),fi5.transpose()),train_target)
w3_0_0 = np.array(train_x).mean()-np.dot(w3_0[1:],np.array([fi5[1].mean(),fi5[2].mean()]))
train_rmse2_0 = rmse(w3_0_0+np.dot(w3_0[1:],fi5.transpose()[1:]),train_target)
train_rmse2_0


# In[11]:

fi5 = np.array([func,func1,func14]) #for attribute2
fi5 = fi5.transpose()
w3_0 = np.dot(np.dot(inv(np.dot(fi5.transpose(),fi5)),fi5.transpose()),train_target)
w3_0_0 = np.array(train_x).mean()-np.dot(w3_0[1:],np.array([fi5[1].mean(),fi5[2].mean()]))
train_rmse2_0 = rmse(w3_0_0+np.dot(w3_0[1:],fi5.transpose()[1:]),train_target)
train_rmse2_0


# In[12]:

fi5 = np.array([func,func1,func17]) #for attribute3
fi5 = fi5.transpose()
w3_0 = np.dot(np.dot(inv(np.dot(fi5.transpose(),fi5)),fi5.transpose()),train_target)
w3_0_0 = np.array(train_x).mean()-np.dot(w3_0[1:],np.array([fi5[1].mean(),fi5[2].mean()]))
train_rmse2_0 = rmse(w3_0_0+np.dot(w3_0[1:],fi5.transpose()[1:]),train_target)
train_rmse2_0


# In[13]:

fi5 = np.array([func,func1,func19]) #for attribute4
fi5 = fi5.transpose()
w3_0 = np.dot(np.dot(inv(np.dot(fi5.transpose(),fi5)),fi5.transpose()),train_target)
w3_0_0 = np.array(train_x).mean()-np.dot(w3_0[1:],np.array([fi5[1].mean(),fi5[2].mean()]))
train_rmse2_0 = rmse(w3_0_0+np.dot(w3_0[1:],fi5.transpose()[1:]),train_target)
train_rmse2_0


# In[ ]:



