# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 16:08:06 2018

@author: mayao
"""
import numpy as np                        #矩阵运算
import random                             #随机生成数据
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] # 使 plt中的文字体能够显现#
plt.rcParams['axes.unicode_minus']=False # 使 plt中的文字体能够显现 #

def genData(rnumber,bias,variance):     #行数，偏置，方差
    x = np.zeros(shape=(rnumber,2))     # 读取行数为numpoints,列数为2的全0矩阵
    y = np.zeros(shape=(rnumber))
    for i in range(0,rnumber):          #从0 到numpoints-1
        x[i][0]=1
        x[i][1]=i
        y[i]=(i+bias)+random.uniform(0,1)+variance
    return (x,y)
# 梯度下降
def gradientDescent(x,y,theta,alpha,m,numIterations):     # 梯度下降 theta需要返回的系数，a是学习率，m是实例个数，,numIterations循环次数
    xTran = np.transpose(x)  #求转置矩阵
    costs = []
    num = []
    for i in range(numIterations):
        hypothesis = np.dot(x,theta)            #dot是两个矩阵相乘
        loss = hypothesis-y
        cost = np.sum(loss**2)/(2*m)
        gradient=np.dot(xTran,loss)/m
        theta = theta-alpha*gradient
        costs.append(cost)
        num.append(i)
        print ("迭代次数 %d | 损失:%f" %(i,cost))
    return (theta,costs,num)
# 调用根数据函数计算得到x和y矩阵
x,y = genData(100, 15, 5)
print ("x:",x)
print ("y:",y)
m,n = np.shape(x)
n_y = np.shape(y)

print("m:"+str(m)+" n:"+str(n)+" n_y:"+str(n_y))#行数，列数，以及输出
# 设置参数
numIterations = 100000
alpha = 0.0005
theta = np.ones(2)     # 设定theta矩阵
print(theta)
theta,costs,num= gradientDescent(x, y, theta, alpha, m, numIterations)
print(theta)


## 结果绘图
plt.figure()
IT = np.linspace(0,numIterations,100)
#绘值公路客运量对比图；
plt.title(u'loss ')
plt.plot( num,costs,'r:o',label=u'')
plt.xlabel (u'迭代次数')
plt.ylabel(u'损失数值 ')
plt.show()
