
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from math import exp
train_data = pd.read_csv('C:\\Users\\gaexp\\OneDrive\\Documents\\Machine Learning\Homework\\[2017]ML_HW1\\data\\4_train.csv')
test_data = pd.read_csv('C:\\Users\\gaexp\\OneDrive\\Documents\\Machine Learning\Homework\\[2017]ML_HW1\\data\\4_test.csv')


# In[2]:

def rmse(predictions, targets):
    return np.sqrt(((np.array(predictions) - np.array(targets)) ** 2).mean())


# In[3]:

func0 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
func0_1 = np.array([1,1,1,1,1,1,1,1,1,1])
func0_2 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
def func(x):
    return x
def func1(x):
    return x**2
def func2(x):
    return x**3
def func3(x):
    return x**4
def func4(x):
    return x**5
def func5(x):
    return x**6
def func6(x):
    return x**7
def func7(x):
    return x**8
def func8(x):
    return x**9


# In[4]:

fi = np.array([func0,func(train_data['x'])])
fi1 = fi.transpose()
w1 = np.dot(np.dot(inv(np.dot(fi1.transpose(),fi1)),fi1.transpose()),np.array(train_data['t']))
fi = np.array([func0,func(train_data['x']),func1(train_data['x'])])
fi1 = fi.transpose()
w2 = np.dot(np.dot(inv(np.dot(fi1.transpose(),fi1)),fi1.transpose()),np.array(train_data['t']))
fi = np.array([func0,func(train_data['x']),func1(train_data['x']),func2(train_data['x'])])
fi1 = fi.transpose()
w3 = np.dot(np.dot(inv(np.dot(fi1.transpose(),fi1)),fi1.transpose()),np.array(train_data['t']))
fi = np.array([func0,func(train_data['x']),func1(train_data['x']),func2(train_data['x']),func3(train_data['x'])])
fi1 = fi.transpose()
w4 = np.dot(np.dot(inv(np.dot(fi1.transpose(),fi1)),fi1.transpose()),np.array(train_data['t']))
fi = np.array([func0,func(train_data['x']),func1(train_data['x']),func2(train_data['x']),func3(train_data['x']),func4(train_data['x'])])
fi1 = fi.transpose()
w5 = np.dot(np.dot(inv(np.dot(fi1.transpose(),fi1)),fi1.transpose()),np.array(train_data['t']))
fi = np.array([func0,func(train_data['x']),func1(train_data['x']),func2(train_data['x']),func3(train_data['x']),func4(train_data['x']),func5(train_data['x'])])
fi1 = fi.transpose()
w6 = np.dot(np.dot(inv(np.dot(fi1.transpose(),fi1)),fi1.transpose()),np.array(train_data['t']))
fi = np.array([func0,func(train_data['x']),func1(train_data['x']),func2(train_data['x']),func3(train_data['x']),func4(train_data['x']),func5(train_data['x']),func6(train_data['x'])])
fi1 = fi.transpose()
w7 = np.dot(np.dot(inv(np.dot(fi1.transpose(),fi1)),fi1.transpose()),np.array(train_data['t']))
fi = np.array([func0,func(train_data['x']),func1(train_data['x']),func2(train_data['x']),func3(train_data['x']),func4(train_data['x']),func5(train_data['x']),func6(train_data['x']),func7(train_data['x'])])
fi1 = fi.transpose()
w8 = np.dot(np.dot(inv(np.dot(fi1.transpose(),fi1)),fi1.transpose()),np.array(train_data['t']))
fi = np.array([func0,func(train_data['x']),func1(train_data['x']),func2(train_data['x']),func3(train_data['x']),func4(train_data['x']),func5(train_data['x']),func6(train_data['x']),func7(train_data['x']),func8(train_data['x'])])
fi1 = fi.transpose()
w9 = np.dot(np.dot(inv(np.dot(fi1.transpose(),fi1)),fi1.transpose()),np.array(train_data['t']))
w1


# In[5]:

w1_0 = np.array(train_data['t']).mean()-np.dot(w1[1:],np.array([func(train_data['x']).mean()]))
w2_0 = np.array(train_data['t']).mean()-np.dot(w2[1:],np.array([func(train_data['x']).mean(),func1(train_data['x']).mean()]))
w3_0 = np.array(train_data['t']).mean()-np.dot(w3[1:],np.array([func(train_data['x']).mean(),func1(train_data['x']).mean(),func2(train_data['x']).mean()]))
w4_0 = np.array(train_data['t']).mean()-np.dot(w4[1:],np.array([func(train_data['x']).mean(),func1(train_data['x']).mean(),func2(train_data['x']).mean(),func3(train_data['x']).mean()]))
w5_0 = np.array(train_data['t']).mean()-np.dot(w5[1:],np.array([func(train_data['x']).mean(),func1(train_data['x']).mean(),func2(train_data['x']).mean(),func3(train_data['x']).mean(),func4(train_data['x']).mean()]))
w6_0 = np.array(train_data['t']).mean()-np.dot(w6[1:],np.array([func(train_data['x']).mean(),func1(train_data['x']).mean(),func2(train_data['x']).mean(),func3(train_data['x']).mean(),func4(train_data['x']).mean(),func5(train_data['x']).mean()]))
w7_0 = np.array(train_data['t']).mean()-np.dot(w7[1:],np.array([func(train_data['x']).mean(),func1(train_data['x']).mean(),func2(train_data['x']).mean(),func3(train_data['x']).mean(),func4(train_data['x']).mean(),func5(train_data['x']).mean(),func6(train_data['x']).mean()]))
w8_0 = np.array(train_data['t']).mean()-np.dot(w8[1:],np.array([func(train_data['x']).mean(),func1(train_data['x']).mean(),func2(train_data['x']).mean(),func3(train_data['x']).mean(),func4(train_data['x']).mean(),func5(train_data['x']).mean(),func6(train_data['x']).mean(),func7(train_data['x']).mean()]))
w9_0 = np.array(train_data['t']).mean()-np.dot(w9[1:],np.array([func(train_data['x']).mean(),func1(train_data['x']).mean(),func2(train_data['x']).mean(),func3(train_data['x']).mean(),func4(train_data['x']).mean(),func5(train_data['x']).mean(),func6(train_data['x']).mean(),func7(train_data['x']).mean(),func8(train_data['x']).mean()]))


# In[6]:

rms_train1 = rmse(w1_0+np.dot(w1[1:],np.array([func(train_data['x'])])),train_data['t'])
rms_train2 = rmse(w2_0+np.dot(w2[1:],np.array([func(train_data['x']),func1(train_data['x'])])),train_data['t'])
rms_train3 = rmse(w3_0+np.dot(w3[1:],np.array([func(train_data['x']),func1(train_data['x']),func2(train_data['x'])])),train_data['t'])
rms_train4 = rmse(w4_0+np.dot(w4[1:],np.array([func(train_data['x']),func1(train_data['x']),func2(train_data['x']),func3(train_data['x'])])),train_data['t'])
rms_train5 = rmse(w5_0+np.dot(w5[1:],np.array([func(train_data['x']),func1(train_data['x']),func2(train_data['x']),func3(train_data['x']),func4(train_data['x'])])),train_data['t'])
rms_train6 = rmse(w6_0+np.dot(w6[1:],np.array([func(train_data['x']),func1(train_data['x']),func2(train_data['x']),func3(train_data['x']),func4(train_data['x']),func5(train_data['x'])])),train_data['t'])
rms_train7 = rmse(w7_0+np.dot(w7[1:],np.array([func(train_data['x']),func1(train_data['x']),func2(train_data['x']),func3(train_data['x']),func4(train_data['x']),func5(train_data['x']),func6(train_data['x'])])),train_data['t'])
rms_train8 = rmse(w8_0+np.dot(w8[1:],np.array([func(train_data['x']),func1(train_data['x']),func2(train_data['x']),func3(train_data['x']),func4(train_data['x']),func5(train_data['x']),func6(train_data['x']),func7(train_data['x'])])),train_data['t'])
rms_train9 = rmse(w9_0+np.dot(w9[1:],np.array([func(train_data['x']),func1(train_data['x']),func2(train_data['x']),func3(train_data['x']),func4(train_data['x']),func5(train_data['x']),func6(train_data['x']),func7(train_data['x']),func8(train_data['x'])])),train_data['t'])


# In[7]:

rms_test1 = rmse(w1_0+np.dot(w1[1:],np.array([func(test_data['x'])])),test_data['t'])
rms_test2 = rmse(w2_0+np.dot(w2[1:],np.array([func(test_data['x']),func1(test_data['x'])])),test_data['t'])
rms_test3 = rmse(w3_0+np.dot(w3[1:],np.array([func(test_data['x']),func1(test_data['x']),func2(test_data['x'])])),test_data['t'])
rms_test4 = rmse(w4_0+np.dot(w4[1:],np.array([func(test_data['x']),func1(test_data['x']),func2(test_data['x']),func3(test_data['x'])])),test_data['t'])
rms_test5 = rmse(w5_0+np.dot(w5[1:],np.array([func(test_data['x']),func1(test_data['x']),func2(test_data['x']),func3(test_data['x']),func4(test_data['x'])])),test_data['t'])
rms_test6 = rmse(w6_0+np.dot(w6[1:],np.array([func(test_data['x']),func1(test_data['x']),func2(test_data['x']),func3(test_data['x']),func4(test_data['x']),func5(test_data['x'])])),test_data['t'])
rms_test7 = rmse(w7_0+np.dot(w7[1:],np.array([func(test_data['x']),func1(test_data['x']),func2(test_data['x']),func3(test_data['x']),func4(test_data['x']),func5(test_data['x']),func6(test_data['x'])])),test_data['t'])
rms_test8 = rmse(w8_0+np.dot(w8[1:],np.array([func(test_data['x']),func1(test_data['x']),func2(test_data['x']),func3(test_data['x']),func4(test_data['x']),func5(test_data['x']),func6(test_data['x']),func7(test_data['x'])])),test_data['t'])
rms_test9 = rmse(w9_0+np.dot(w9[1:],np.array([func(test_data['x']),func1(test_data['x']),func2(test_data['x']),func3(test_data['x']),func4(test_data['x']),func5(test_data['x']),func6(test_data['x']),func7(test_data['x']),func8(test_data['x'])])),test_data['t'])


# In[8]:

train_rms =pd.DataFrame({'M':[1,2,3,4,5,6,7,8,9],'RMSE':[rms_train1,rms_train2,rms_train3,rms_train4,rms_train5,rms_train6,rms_train7,rms_train8,rms_train9]})
test_rms =pd.DataFrame({'M':[1,2,3,4,5,6,7,8,9],'RMSE':[rms_test1,rms_test2,rms_test3,rms_test4,rms_test5,rms_test6,rms_test7,rms_test8,rms_test9]})
print(train_rms)
print(test_rms)


# In[9]:

plt.plot(train_rms['M'],train_rms['RMSE'],'b-o',label='Training')
plt.plot(test_rms['M'],test_rms['RMSE'],'r-o',label='Test')
plt.xlabel ('M')
plt.ylabel ('RMS')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# In[10]:

x = np.linspace(0,6.5,50)
plt.figure(figsize=(10,40))
plt.subplot(911)
plt.scatter(train_data['x'],train_data['t'], label='train')
plt.scatter(test_data['x'],test_data['t'], label='test')
plt.plot(x,w1_0+np.dot(w1[1:],np.array([x])),'r--',label='fit')
plt.title('M=1')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.subplot(912)
plt.scatter(train_data['x'],train_data['t'], label='train')
plt.scatter(test_data['x'],test_data['t'], label='test')
plt.plot(x,w2_0+np.dot(w2[1:],np.array([x,x**2])),'g--',label='fit')
plt.title('M=2')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.subplot(913)
plt.scatter(train_data['x'],train_data['t'], label='train')
plt.scatter(test_data['x'],test_data['t'], label='test')
plt.plot(x,w3_0+np.dot(w3[1:],np.array([x,x**2,x**3])),'b--',label='fit')
plt.title('M=3')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.subplot(914)
plt.scatter(train_data['x'],train_data['t'], label='train')
plt.scatter(test_data['x'],test_data['t'], label='test')
plt.plot(x,w4_0+np.dot(w4[1:],np.array([x,x**2,x**3,x**4])),'c--',label='fit')
plt.title('M=4')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.subplot(915)
plt.scatter(train_data['x'],train_data['t'], label='train')
plt.scatter(test_data['x'],test_data['t'], label='test')
plt.plot(x,w5_0+np.dot(w5[1:],np.array([x,x**2,x**3,x**4,x**5])),'m--',label='fit')
plt.title('M=5')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.subplot(916)
plt.scatter(train_data['x'],train_data['t'], label='train')
plt.scatter(test_data['x'],test_data['t'], label='test')
plt.plot(x,w6_0+np.dot(w6[1:],np.array([x,x**2,x**3,x**4,x**5,x**6])),'y--',label='fit')
plt.title('M=6')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.subplot(917)
plt.scatter(train_data['x'],train_data['t'], label='train')
plt.scatter(test_data['x'],test_data['t'], label='test')
plt.plot(x,w7_0+np.dot(w7[1:],np.array([x,x**2,x**3,x**4,x**5,x**6,x**7])),'k--',label='fit')
plt.title('M=7')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.subplot(918)
plt.scatter(train_data['x'],train_data['t'], label='train')
plt.scatter(test_data['x'],test_data['t'], label='test')
plt.plot(x,w8_0+np.dot(w8[1:],np.array([x,x**2,x**3,x**4,x**5,x**6,x**7,x**8])),'r--',label='fit')
plt.title('M=8')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.subplot(919)
plt.scatter(train_data['x'],train_data['t'], label='train')
plt.scatter(test_data['x'],test_data['t'], label='test')
plt.plot(x,w9_0+np.dot(w9[1:],np.array([x,x**2,x**3,x**4,x**5,x**6,x**7,x**8,x**9])),'g--',label='fit')
plt.title('M=9')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# In[11]:

def rmse1(predictions, targets,m,w):
    return np.sqrt(((np.array(predictions) - np.array(targets)) ** 2+m*np.dot(w,w.transpose())).mean())


# In[12]:

rms_train = []
landa = exp(-20)
for m in range(20):
    fi = np.array([func0,func(train_data['x']),func1(train_data['x']),func2(train_data['x']),func3(train_data['x']),func4(train_data['x']),func5(train_data['x']),func6(train_data['x']),func7(train_data['x']),func8(train_data['x'])])
    fi1 = fi.transpose()
    w9 = np.dot(np.dot(inv(landa*np.identity(10)+np.dot(fi1.transpose(),fi1)),fi1.transpose()),np.array(train_data['t']))
    w9_0 = np.array(train_data['t']).mean()-np.dot(w9[1:],np.array([func(train_data['x']).mean(),func1(train_data['x']).mean(),func2(train_data['x']).mean(),func3(train_data['x']).mean(),func4(train_data['x']).mean(),func5(train_data['x']).mean(),func6(train_data['x']).mean(),func7(train_data['x']).mean(),func8(train_data['x']).mean()]))
    rms_train.append(rmse1(w9_0+np.dot(w9[1:],np.array([func(train_data['x']),func1(train_data['x']),func2(train_data['x']),func3(train_data['x']),func4(train_data['x']),func5(train_data['x']),func6(train_data['x']),func7(train_data['x']),func8(train_data['x'])])),train_data['t'],landa,w9))
    landa = landa * exp(1)


# In[13]:

rms_test = []
landa = exp(-20)
for m in range(20):
    fi = np.array([func0,func(train_data['x']),func1(train_data['x']),func2(train_data['x']),func3(train_data['x']),func4(train_data['x']),func5(train_data['x']),func6(train_data['x']),func7(train_data['x']),func8(train_data['x'])])
    fi1 = fi.transpose()
    w9 = np.dot(np.dot(inv(landa*np.identity(10)+np.dot(fi1.transpose(),fi1)),fi1.transpose()),np.array(train_data['t']))
    w9_0 = np.array(train_data['t']).mean()-np.dot(w9[1:],np.array([func(train_data['x']).mean(),func1(train_data['x']).mean(),func2(train_data['x']).mean(),func3(train_data['x']).mean(),func4(train_data['x']).mean(),func5(train_data['x']).mean(),func6(train_data['x']).mean(),func7(train_data['x']).mean(),func8(train_data['x']).mean()]))
    rms_test.append(rmse1(w9_0+np.dot(w9[1:],np.array([func(test_data['x']),func1(test_data['x']),func2(test_data['x']),func3(test_data['x']),func4(test_data['x']),func5(test_data['x']),func6(test_data['x']),func7(test_data['x']),func8(test_data['x'])])),test_data['t'],landa,w9))
    landa = landa * exp(1)


# In[14]:

a=[]
for i in range(20):
    a.append(i-20)
train_rms1 =pd.DataFrame({'ln(landa)':a,'RMSE':rms_train})
test_rms1 = pd.DataFrame({'ln(landa)':a,'RMSE':rms_test})
plt.plot(train_rms1['ln(landa)'],train_rms1['RMSE'],'r-',label='train')
plt.plot(test_rms1['ln(landa)'],test_rms1['RMSE'],'b-',label='test')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('ln(m)')
plt.ylabel('RMSE')
plt.xticks([-20,-15,-10,-5,0])
plt.show()


# In[ ]:



