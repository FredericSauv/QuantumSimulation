#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:27:22 2019

@author: fred
Mosly from GPyOpt

"""


from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize 
from matplotlib.lines import Line2D 

def reshape(x,input_dim):
    x = np.array(x)
    if x.size ==input_dim:
        x = x.reshape((1,input_dim))
    return x


class function2d:
    def plot(self):
        bounds = self.bounds
        x1 = np.linspace(bounds[0][0], bounds[0][1], 100)
        x2 = np.linspace(bounds[1][0], bounds[1][1], 100)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.hstack((X1.reshape(100*100,1),X2.reshape(100*100,1)))
        Y = self.f(X)
        
        #fig = plt.figure()
        #ax = fig.gca(projection='3d')
        #ax.plot_surface(X1, X2, Y.reshape((100,100)), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        #ax.zaxis.set_major_locator(LinearLocator(10))
        #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        #ax.set_title(self.name)    
            
        plt.figure()    
        plt.contourf(X1, X2, Y.reshape((100,100)),100)
        if (len(self.min)>1):    
            plt.plot(np.array(self.min)[:,0], np.array(self.min)[:,1], 'w.', markersize=20, label=u'Observations')
        else:
            plt.plot(self.min[0][0], self.min[0][1], 'w.', markersize=20, label=u'Observations')
        plt.colorbar()
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title(self.name)
        plt.show()
        
        
        
class branin(function2d):
    '''
    Branin function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,a=None,b=None,c=None,r=None,s=None,t=None,sd=None):
        self.input_dim = 2
        if bounds is  None: self.bounds = [(-5,10),(1,15)]
        else: self.bounds = bounds
        if a==None: self.a = 1
        else: self.a = a           
        if b==None: self.b = 5.1/(4*np.pi**2)
        else: self.b = b
        if c==None: self.c = 5/np.pi
        else: self.c = c
        if r==None: self.r = 6
        else: self.r = r
        if s==None: self.s = 10 
        else: self.s = s
        if t==None: self.t = 1/(8*np.pi)
        else: self.t = t    
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.min = [(-np.pi,12.275),(np.pi,2.275),(9.42478,2.475)] 
        self.fmin = 0.397887
        self.name = 'Branin'
        self.x_seen = []
    
    
    def f(self,X):
        self.x_seen.append(X)
        X = reshape(X,self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim: 
            return 'Wrong input dimension'  
        else:
            x1 = X[:,0]
            x2 = X[:,1]
            fval = self.a * (x2 - self.b*x1**2 + self.c*x1 - self.r)**2 + self.s*(1-self.t)*np.cos(x1) + self.s 
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return fval.reshape(n,1) + noise





#nm
#  scipy.optimize.minimize(fun, x0, args=(), method='Nelder-Mead', tol=None, callback=None, options={'func': None, 'maxiter': None, 'maxfev': None, 'disp': False, 'return_all': False, 'initial_simplex': None, 'xatol': 0.0001, 'fatol': 0.0001, 'adaptive': False})
br = branin()
f_min = lambda x: br.f(x)
#x0 = np.array([[np.random.uniform(b[0], b[1]) for b in br.bounds] for _ in range(3)])
x0 = np.array([(6.3, 14.5), (6.57, 11.12), (2.29, 14.303)])
optim_nm = minimize(f_min, x0[0],method='Nelder-Mead', options={'initial_simplex':x0, 'return_all':True, 'xatol': 1,'fatol':0.01})

all_points_nm = np.array(br.x_seen)
print(len(all_points_nm))
nb_points = len(all_points_nm)
fig = plt.figure()
ax = fig.add_subplot(111)
x = all_points_nm[:,0]
y = all_points_nm[:,1]
to_plot = [35,36, 37]
for i in range(nb_points-3):
    if(i in to_plot):
        x_tmp = [x[i], x[i+1], x[i+2], x[i]]
        y_tmp = [y[i], y[i+1], y[i+2], y[i]]
        line = Line2D(x_tmp, y_tmp)
        ax.add_line(line)


ax.set_xlim(-5, 10)
ax.set_ylim(1, 15)

br.plot()
