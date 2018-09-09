#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 19:06:58 2018

@author: fred
"""

import GPy
import numpy as np
from scipy.stats import multivariate_normal as mv
import matplotlib.pylab as plt

plt.rc('text', usetex=True)
#==============================================================================
# ***GOAL Present basis of GP 
#==============================================================================
nb_points_path = 50
nb_func = 50

l = 0.2
k = GPy.kern.RBF(input_dim=1,lengthscale = l, variance = 1.0)
X = np.linspace(0.,1., nb_points_path)
X = X[:,None]
mu = np.zeros((nb_points_path))
C = k.K(X,X)
Z = np.squeeze(np.random.multivariate_normal(mu, C, nb_func))
mv0 = mv(mean = mu, cov = C , allow_singular=True)
log_p = mv0.logpdf(Z)
tmp =  np.power(log_p / np.sum(log_p), 8)
scale = 1.2 /np.max(tmp)
p_norm = scale * tmp
for n, zz in enumerate(Z):
    plt.plot(X[:], zz, color = 'blue', linewidth = p_norm[n])
plt.savefig('prior_more_3.pdf')
#plot one func = 
nb_1 = 3
x_fin = X[-1,0]

x_lim = [0, 1.35]
y_lim = [-2, 2]

arbitray_labels = [r"p(f)=0.1", r"p(f)=0.3", r"p(f)=0.6"]

for n in np.arange(nb_1):
    plt.plot(X[:], Z[n], color='blue')
    z_fin = Z[n, -1]
    plt.text(x_fin+0.02, z_fin +0.02, arbitray_labels[n], fontsize=15, color='b')
plt.xlim(x_lim)
plt.ylim(y_lim)


for n, zz in enumerate(Z):
    plt.plot(X[:], zz, color = 'blue', linewidth = p_norm[n])

#plt.text(x_fin+0.02, 0, r"$p(f) \approx \mathcal{GP}$", fontsize=15, color='b')


### Likelihood
nb_data = 5
# fake_data = Z[0,:] + np.random.normal(0, 0.4, nb_points_path)
fake_data = np.random.normal(0, 1, nb_points_path)
index_random = np.random.choice(nb_points_path, nb_data)
X_chosen = X[index_random]
fake_data_chosen = fake_data[index_random]
font = ['r--', 'g--', 'm--']
offset = [-0.01, 0, 0.01]
arbitray_labels = [r"$\textbf{p(D | f)=0.12}$", r"p(f)=0.3", r"p(f)=0.6"]


#for i in index_random:
#    x = X[i,0]
#    plt.plot([x, x], [fake_data[i], Z_index[i]], 'r--')
#plt.text(x_fin+0.02, z_fin +0.02, r'$p(\mathcal{D}|f) = 0.03$', fontsize=15, color='b')
#plt.legend(loc=4, fontsize =15)
#

eps = Z[:, index_random] - fake_data_chosen 

logL = np.sum(- np.square(eps)/(2* np.sqrt(0.4)),1)
width = -0.8/logL * 2
    
for n, zz in enumerate(Z):
    plt.plot(X[:], zz, color = 'blue', linewidth = width[n])
plt.scatter(X[index_random], fake_data[index_random], color='r', s=60, marker='s', alpha=.9, label=r'Observations')
plt.legend(loc=4, fontsize=12)
plt.savefig('likelihood_more_3.pdf')


plt.ylim(y_lim)

##add dotted line
plt.plot(X[:], Z_index, color='blue')
plt.xlim([0,1.39])
plt.scatter(X[index_random], fake_data[index_random], color='r', s=60, marker='s', alpha=.4)
plt.xlim(x_lim)
plt.ylim(y_lim)
plt.text(x_fin+0.02, z_fin +0.02, r"p(f|\mathcal{D}) = 0.03", fontsize=12, color='b')


X_reg = X_chosen.reshape((len(X_chosen), 1))
Y_reg = fake_data_chosen.reshape((len(fake_data_chosen),1))


kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=0.2)
model = GPy.models.GPRegression(X_reg, Y_reg, kernel)
model.constrain_positive()
model.optimize()
model.plot()


m, v = model.predict(X)
lower = m - 1.96*np.sqrt(v)
upper = m + 1.96*np.sqrt(v)
plt.plot(X[:], m, 'b', label=r"$\mu(x) = E_{f| \mathcal{D}}[f(x)] $")
plt.plot(X[:], lower, 'b', linewidth=0.2)
plt.plot(X[:], upper, 'b', linewidth=0.2)
plt.fill_between(np.squeeze(X), np.squeeze(upper), np.squeeze(lower), alpha=0.2,label=r"$\mu(x) + 1.96 \sigma(X)$")
plt.scatter(X_reg, Y_reg, color='r', s=60, marker='s', alpha=.9, label = r"$Observations$" )
plt.legend(fontsize=14)
plt.savefig('fitted_2.pdf')

X_new = np.linspace(0.,1., nb_points_path*10)
X_new = X_new[:,None]
ps = model.posterior_samples_f(X_new, 50)
for n, zz in enumerate(np.transpose(ps)):
    plt.plot(X_new[:], zz, color = 'blue', linewidth = 1)
plt.scatter(X_reg, Y_reg, color='r', s=60, marker='s', alpha=.9, label = r"$Observations$")
plt.legend(loc=4, fontsize=12)
plt.savefig('posterior_more_2.pdf')
model.plot()

### Look at kernel
k = GPy.kern.rbf(input_dim=1,lengthscale=0.2)
X = np.linspace(0.,1.,500)
# 500 points evenly spaced over [0,1]
X = X[:,None]
# reshape X to make it n*D
mu = np.zeros((500))
# vector of the means
C = k.K(X,X)
# covariance matrix
# Generate 20 sample path with mean mu and covariance C
Z = np.random.multivariate_normal(mu,C,20)
pb.figure()
# open new plotting window
for i in range(20):
pb.plot(X[:],Z[i,:])