#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 14:43:27 2019

https://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/multiple%20outputs.ipynb
"""
import numpy as np
import matplotlib.pyplot as plt
import GPy
import scipy.io
import copy
import time
from IPython.display import display

data = scipy.io.loadmat('olympics.mat')


map_task = {'male100':0, 'female100':1, 'male200':2, 'female200':3, 'male400':4, 'female400':5}
years = np.array([ 1896. ,  1900. ,  1904. ,  1906. ,  1908. ,  1912. ,  1920. ,  1924. ,  1928. ,  1932. , 1936. ,  1948. ,  1952. ,  1956. ,  1960. ,  1964. ,  1968. ,  1972. ,  1976. ,  1980. ,
  1984. ,  1988. ,  1992. ,  1996. ,  2000. ,  2004. ,  2008.])


list_X = list()
list_Y = list()
for n, task in enumerate(map_task):
    index_task = map_task[task]
    data_tmp = data[task]
    mask = np.array([y in years for y in data_tmp[:, 0]])
    Y_tmp = data_tmp[:,1][:, np.newaxis]
    X_tmp = np.hstack((data_tmp[mask, 0][:, np.newaxis], np.repeat(index_task, np.sum(mask))[:, np.newaxis]))
    if(n == 0):
        X = X_tmp
        y = Y_tmp
    else:
        X = np.vstack((X, X_tmp))
        y = np.vstack((y, Y_tmp))
        



print('First column of X contains the olympic years.')
print(np.unique(X[:, 0]))
print('Second column of X contains the event index.')
print(np.unique(X[:, 1]))



markers = ['bo', 'ro', 'bx', 'rx', 'bs', 'rs']
for i in range(6):
    # extract the event 
    x_event = X[np.nonzero(X[:, 1]==i), 0]
    y_event = y[np.nonzero(X[:, 1]==i), 0]
    plt.plot(x_event, y_event, markers[i])
plt.title('Olympic Sprint Times')
plt.xlabel('year')
plt.ylabel('time/s')



##RANK 0
kernX = GPy.kern.Matern52(1, variance=1., ARD=False)
kernX += GPy.kern.White(1, 0.01)
coreg = GPy.kern.Coregionalize(1, output_dim=6, rank=0)
coreg.kappa.fix()
kern = kernX ** coreg

print(kern)
#print(kern.coregion.W[:])
model = GPy.models.GPRegression(X, y, kern)
ts = time.time()
model.optimize()
t_rank0 = time.time() - ts
fig, ax = plt.subplots()
for i in range(6):
    model.plot(fignum=1,fixed_inputs=[(1, i)],ax=ax,legend=i==0)
plt.xlabel('years')
plt.ylabel('time/s')

##RANK 1
kern = GPy.kern.RBF(1, lengthscale=80) ** GPy.kern.Coregionalize(1, output_dim=6, rank=1)
print(kern)
#print(kern.coregion.W[:])
model = GPy.models.GPRegression(X, y, kern)
ts = time.time()
model.optimize()
t_rank0 = time.time() - ts
fig, ax = plt.subplots()
for i in range(6):
    model.plot(fignum=1,fixed_inputs=[(1, i)],ax=ax,legend=i==0)
plt.xlabel('years')
plt.ylabel('time/s')


##Engineered kernels
kern1 = GPy.kern.RBF(1, lengthscale=80)**GPy.kern.Coregionalize(1,output_dim=6, rank=1)
kern2 = GPy.kern.Bias(1)**GPy.kern.Coregionalize(1,output_dim=6, rank=1)
kern = kern1 + kern2

model = GPy.models.GPRegression(X, y, kern)
model.optimize()

fig, ax = plt.subplots()
for i in range(6):
    model.plot(fignum=1,fixed_inputs=[(1, i)],ax=ax,legend=i==0)
plt.xlabel('years')
plt.ylabel('time/s')



