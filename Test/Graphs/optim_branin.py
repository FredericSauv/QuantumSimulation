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
from scipy.optimize import minimize, differential_evolution
from matplotlib.lines import Line2D 

def reshape(x,input_dim):
    x = np.array(x)
    if x.size ==input_dim:
        x = x.reshape((1,input_dim))
    return x


class function2d:
    def plot(self, infos=True):
        bounds = self.bounds
        x1 = np.linspace(bounds[0][0], bounds[0][1], 100)
        x2 = np.linspace(bounds[1][0], bounds[1][1], 100)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.hstack((X1.reshape(100*100,1),X2.reshape(100*100,1)))
        Y = self.f(X, append=False)
        
        #fig = plt.figure()
        #ax = fig.gca(projection='3d')
        #ax.plot_surface(X1, X2, Y.reshape((100,100)), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        #ax.zaxis.set_major_locator(LinearLocator(10))
        #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        #ax.set_title(self.name)    


        fig = plt.figure()    
        ax = fig.add_subplot(111)   
        cs = ax.contourf(X1, X2, Y.reshape((100,100)),100, cmap=cm.PuBu_r)
        alpha_val = 1      

        if infos:
            if (len(self.min)>1):    
                ax.plot(np.array(self.min)[:,0], np.array(self.min)[:,1], 'r.', markersize=10, alpha = alpha_val, label=u'Minima')
            else:
                ax.plot(self.min[0][0], self.min[0][1], 'r.', markersize=10, alpha = alpha_val, label=u'Minima')
            fig.colorbar(cs)
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            #fig.suptitle(self.name)
        else:
            if (len(self.min)>1):    
                ax.plot(np.array(self.min)[:,0], np.array(self.min)[:,1], 'r.', markersize=10, alpha = alpha_val)
            else:
                ax.plot(self.min[0][0], self.min[0][1], 'r.', markersize=10, alpha = 0.2)
        fig.show()
        
        return fig, ax
        
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
    
    
    def f(self,X, append = True):
        if append:
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




###############################################################################
#               NEDLER MEAD
#  scipy.optimize.minimize(fun, x0, args=(), method='Nelder-Mead', tol=None, callback=None, 
#options={'func': None, 'maxiter': None, 'maxfev': None, 'disp': False, 'return_all': False, 
#'initial_simplex': None, 'xatol': 0.0001, 'fatol': 0.0001, 'adaptive': False})
###############################################################################
#optim
br = branin()
f_min = lambda x: br.f(x)
#x0 = np.array([[np.random.uniform(b[0], b[1]) for b in br.bounds] for _ in range(3)])
x0 = np.array([(6.3, 14.5), (6.57, 11.12), (2.29, 14.303)])
optim_nm = minimize(f_min, x0[0],method='Nelder-Mead', options={'initial_simplex':x0, 'return_all':True, 'xatol': 0.01,'fatol':0.01})


#plot res
all_points_nm = np.array(br.x_seen)
print(len(all_points_nm))
nb_points = len(all_points_nm)


fig, ax = br.plot()
x = all_points_nm[:,0]
y = all_points_nm[:,1]
to_plot_main = [0, 1, ]
to_plot_sec = [3, 4, 6, 9, 35,36,37,38]

for i in range(nb_points-3):
    if(i == 0):
        x_tmp = [x[i], x[i+1], x[i+2], x[i]]
        y_tmp = [y[i], y[i+1], y[i+2], y[i]]
        line = Line2D(x_tmp, y_tmp, marker ='s',color='black',linewidth=1., label=r'$1^{st} \; simplex$')
        ax.add_line(line)
    if(i == 1):
        x_tmp = [x[i], x[i+1], x[i+2], x[i]]
        y_tmp = [y[i], y[i+1], y[i+2], y[i]]
        line = Line2D(x_tmp, y_tmp, marker ='x',color='black',linestyle='dashed',linewidth=1, label=r'$2^{nd} simplex$')
        ax.add_line(line)

    elif(i % 2 ==0):
            x_tmp = [x[i], x[i+1], x[i+2], x[i]]
            y_tmp = [y[i], y[i+1], y[i+2], y[i]]
x = all_points_nm[:,0]
y = all_points_nm[:,1]
to_plot_main = [0, 1, ]
to_plot_sec = [3, 4, 6, 9, 35,36,37,38]

for i in range(nb_points-3):
    if(i == 0):
        x_tmp = [x[i], x[i+1], x[i+2], x[i]]
        y_tmp = [y[i], y[i+1], y[i+2], y[i]]
        line = Line2D(x_tmp, y_tmp, marker ='s',color='black',linewidth=1., label=r'$1^{st} \; simplex$')
        ax.add_line(line)
    if(i == 1):
        x_tmp = [x[i], x[i+1], x[i+2], x[i]]
        y_tmp = [y[i], y[i+1], y[i+2], y[i]]
        line = Line2D(x_tmp, y_tmp, marker ='x',color='black',linestyle='dashed',linewidth=1, label=r'$2^{nd} simplex$')
        ax.add_line(line)

    elif(i % 2 ==0):
            x_tmp = [x[i], x[i+1], x[i+2], x[i]]
            y_tmp = [y[i], y[i+1], y[i+2], y[i]]
            line = Line2D(x_tmp, y_tmp, linestyle='dashed', linewidth=0.2, color='black', marker ='x', markersize=0.6)
            ax.add_line(line)
ax.set_xlim(-5, 10)
ax.set_ylim(1, 15)
ax.scatter(all_points_nm[-1][0], all_points_nm[-1][1], c ='yellow', s = 45, marker = '^', label=r'$Final$')
ax.legend(loc='lower left', fontsize=8)
plt.savefig("optim_graphs_nm.pdf", bbox_inches='tight', transparent=True, pad_inches=0)



###############################################################################
#               DE
#scipy.optimize.differential_evolution(func, bounds, args=(), strategy='best1bin', 
#maxiter=None, popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None, 
#callback=None, disp=False, polish=True, init='latinhypercube')
###############################################################################
#optim
br = branin()
f_min = lambda x: br.f(x)
#x0 = np.array([[np.random.uniform(b[0], b[1]) for b in br.bounds] for _ in range(3)])
optim_de = differential_evolution(f_min, br.bounds, popsize = 5, disp = True)

#plot res
all_points_de = np.array(br.x_seen)
print(len(all_points_de))
nb_points = len(all_points_de)

fig, ax = br.plot(infos=False)

x = all_points_de[:,0]
y = all_points_de[:,1]
cmap = cm.get_cmap('plasma')
size= 35
ax.scatter(x[0:10], y[0:10], s=size, c='black',marker='s', label=r'$1^{st}\;gen.$')
ax.scatter(x[10:20], y[10:20], s=size, c='black', marker='o', label=r'$2^{nd}\;gen.$')
ax.scatter(x[70:80], y[70:80], s=size, c='black',marker='x', label=r'$7^{th}\;gen.$')
ax.scatter(x[140:155], y[140:155], s=45, c='yellow', marker='^', label=r'$Final$')
ax.set_xlim(-5, 10)
ax.set_ylim(1, 15)
ax.legend(loc='lower left', fontsize=9)
plt.savefig("optim_graphs_de.pdf", bbox_inches='tight', transparent=True, pad_inches=0)

###############################################################################
#               FGBS
#  scipy.optimize.minimize(fun, x0, args=(), method='L-BFGS-B', jac=None, 
# bounds=None, tol=None, callback=None, options={'disp': None, 'maxcor': 10, 
# 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 
# 'maxiter': 15000, 'iprint': -1, 'maxls': 20})
###############################################################################
#optim
br = branin()
f_min = lambda x: br.f(x)
x0 = np.array([[np.random.uniform(b[0], b[1]) for b in br.bounds] for _ in range(3)])
x0 = [6.3, 14.5]
#x0= [-4.7,1.7]
#x0 = np.array([np.random.uniform(b[0], b[1]) for b in br.bounds])
optim_bfgs = minimize(f_min, x0=x0, bounds = br.bounds, method='L-BFGS-B',options={'disp': True,'ftol': 2.2204e-03, 'gtol': 1e-03})
all_points_gd = np.array(br.x_seen)
print(len(all_points_gd))
nb_points = len(all_points_gd)




x = all_points_gd[:,0]
y = all_points_gd[:,1]
gd_points = np.arange(nb_points//3) *3 + 1
gd_points = np.array([ 1,  4,   10, 13, 16, 19])
for i in range(len(gd_points)-1):
    ind = gd_points[i]
    ind_next = gd_points[i+1] 
    print(x[ind], y[ind])
    ax.arrow(x[ind], y[ind], 0.95 * (x[ind_next]-x[ind]), 0.95 *  (y[ind_next]-y[ind]), width=0.01, head_width=0.25, color='black')
    plt.scatter(x[ind], y[ind], marker='x', s=35, c = 'black')
plt.scatter(all_points_gd[-1][0], all_points_gd[-1][1], c ='yellow', marker = '^')


ax.set_xlim(-5, 10)
ax.set_ylim(1, 15)
#ax.legend(loc='upper left')


scale = [0.95, 0.60, 0.75, 0.75, 0.75]

fake_data = np.array([[6.30000001, 14.5],[7.43089982823581, 1.1],[8.668840930371791, 1.1],[9.9,2.3813562802149137],[9.406160849428316,2.6684957148002137], [9.42535787, 2.4756237 ]])
x= fake_data[:,0]
y= fake_data[:,1]
fig, ax = br.plot(infos=False)
for i in range(len(fake_data)-1):
    ax.arrow(x[i], y[i], scale[i]*(x[i+1]-x[i]), scale[i]*(y[i+1]-y[i]), width=0.01, head_width=0.25, color='black')
    plt.scatter(x[i], y[i], marker='x', s=35, c = 'black')
plt.scatter(all_points_gd[-1][0], all_points_gd[-1][1], c ='yellow', marker = '^', s=45)
ax.set_xlim(-5, 10)
ax.set_ylim(1, 15)
plt.savefig("optim_graphs_lbfgsb.pdf", bbox_inches='tight', transparent=True, pad_inches=0)
