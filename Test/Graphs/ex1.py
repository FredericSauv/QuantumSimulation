#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 13:46:15 2018

@author: frederic
"""
import numpy as np
import numpy.random as rdm
import scipy.special as special
import matplotlib.pylab as plt
import GPy
import scipy.stats as stats


### ============================================================ ###
# Utilities
### ============================================================ ###
# LANDSCAPE: case where a projective measurement measurement 
#            onto the target state is possible
def proba(x):
    """ underlying probability p(|tgt>) """
    x_noise = +np.sin(3*(x+0.3))/2 + (x+0.3)/1.5
    return np.square(np.sin(x_noise))

def counts(x, underlying_p = proba, nb_measures = 1):
    """ Borns rule"""
    return rdm.binomial(nb_measures, underlying_p(x))

def frequency(x, underlying_p = proba, nb_measures = 1):
    """ Frequency of positive outcome"""
    return rdm.binomial(nb_measures, underlying_p(x))/nb_measures


# GP UTILITY function how to use a GP to makes prediction and to compute the 
#            the maximum of the acquisition fucntion  
def invcdf(y):
    """ inverse of the cumulative distribution function"""
    return  np.sqrt(2) * special.erfinv(2*y-1) 

def change_of_var_Phi(m, s):
    """ from arrays of mean and std of NORMAL variables X, produce the mean and std of
    Phi(X) where Phi is the normal cumulative distribution
    E[Phi(X)] = Phi(m/sqrt(1 + s^2))
    Var[Phi(X)] = E - E^2 - T (m/sqrt(1 + s^2), m/sqrt(1 + 2*s^2))
    """
    a = m / np.sqrt(1 + np.square(s))
    h = 1 / np.sqrt(1 + 2 * np.square(s))
    m_cdf = stats.norm.cdf(a)
    s_cdf = np.sqrt(m_cdf - np.square(m_cdf) - 2 * special.owens_t(a, h))
    return m_cdf, s_cdf

def predict(gpmodel, X):
    """ Prediction of the mean and standard deviations for an array of X
    + if the gp has a binomial likelihood (that is g), then prediction are
    done for f = Phi(g) that is a change of variable is needed
    + if not no change of varaiables are required
    """
    mean, var = gpmodel._raw_predict(X, full_cov=False, kern=None)
    std = np.sqrt(var)
    if  type(gpmodel.likelihood) == GPy.likelihoods.Binomial:
        mean, std = change_of_var_Phi(mean, std)
    return mean, std

def acquisition_function(gpmodel, x, w=4):
    """ Compute the acquisition function for a given array of x.
    acq(x) = m(x) + w s(x) where m and s are the mean and standard deviation
    predicted at location x by the model, and w is a weight which can be tuned"""
    m, s = predict(gpmodel, x)
    acq = (m + w * s)
    return acq


def max_acquisition(gpmodel, w=4, alpha=1, domain=(0,4)):
    """ Return the location of the maximum found over a set of random x 
    within the domain"""
    x_rdm = np.random.uniform(low = domain[0], high = domain[1], size = 1000)
    acq = acquisition_function(gpmodel, x_rdm)
    x_next = x_rdm[np.argmax(acq)]
    return x_next

def visualize(func_landscape, gpmodel, x_plot, x_init, y_init):
    """ 2 panels plot
    Top: model/observations/true landscape(F)
    Bottom Acquisition function
    """
    m_plot, s_plot = predict(gpmodel,x_plot)
    F_plot = func_landscape(x_plot)
    fig, (ax1, ax2) = plt.subplots(2, 1,gridspec_kw={'hspace': 0.3,'height_ratios': [3, 1]})
    ax1.plot(x_plot, m_plot, color = col_custom, linewidth = 0.8, label = 'model')
    ax1.fill_between(np.squeeze(x_plot), np.squeeze(m_plot - 1.96 * s_plot), 
                     np.squeeze(m_plot + 1.96 * s_plot), alpha = 0.3, label='model-confidence')
    ax1.plot(x_plot, F_plot, 'r--', linewidth = 0.8, label = 'F')
    ax1.scatter(x_init, y_init, label = 'Observations', marker = 'o', c='red', s=5)
    ax1.legend()
    acq_plot = acquisition_function(gpmodel, x_plot)
    ax2.plot(x_plot, acq_plot, 'r')
    
    


### ============================================================ ###
# Setting up the problem
### ============================================================ ###
SEED = 2956180 #
SEED = np.random.randint(1,1000000000)
np.random.seed(SEED)
NB_SHOTS = 10 #single shot
col_custom = (0.1, 0.2, 0.5)


# function to maximize and over which domain
func = lambda x: counts(x, nb_measures=NB_SHOTS) # single-shot measurement
domain = (0,4)

## Generate some data
nb_init = 30
x_plot = np.linspace(*domain, 1000)[:, np.newaxis]
#x_plot = np.linspace(*(-10, 10), 1000)[:, np.newaxis]
x_init = np.linspace(*domain, nb_init)[:, np.newaxis] #
x_init = rdm.uniform(*domain, nb_init)[:, np.newaxis]
y_init = func(x_init)


## Define the Model used: a GP with an Exponential kernel + Binomial likelihood 
## + Laplace approximation
# specify the kernel k and the value of its initial hyperparameters
k = GPy.kern.Matern52(input_dim = 1, variance = 1., lengthscale = (domain[1]-domain[0])/20)
k.variance.constrain_bounded(1e-2, 5)
k.lengthscale.constrain_bounded(1e-2, 5)

# specify the inference method to use (Laplace approximation)
i_meth = GPy.inference.latent_function_inference.Laplace()

# specify the likelihood
lik = GPy.likelihoods.Binomial()

# Create the GP model
g = GPy.core.GP(X=x_init, Y=y_init, kernel=k, inference_method=i_meth, 
        likelihood=lik, Y_metadata = {'trials':np.ones_like(y_init) * NB_SHOTS})

# fit the hyperparameters
_ = g.optimize_restarts(num_restarts=5) 

visualize(proba, g, x_plot, x_init, y_init)
#plt.savefig("gaussian_1000obs.pdf", bbox_inches='tight', transparent=True, pad_inches=0)


### ============================================================ ###
# Use of a mean function
### ============================================================ ###
MF_VAL = 1.64 #
mf = GPy.core.Mapping(input_dim=1, output_dim=1)
mf.f = lambda x: np.array([[MF_VAL] for xx in np.atleast_2d(x)])
mf.update_gradients = lambda a,b: 0
mf.gradients_X = lambda a,b: 0
# Create the GP model
gmf = GPy.core.GP(X=x_init, Y=y_init, kernel=k, inference_method=i_meth, 
        likelihood=lik, Y_metadata = {'trials':np.ones_like(y_init) * NB_SHOTS},
        mean_function=mf)

# fit the hyperparameters
_ = gmf.optimize_restarts(num_restarts=5) 

visualize(proba, gmf, x_plot, x_init, y_init)
#plt.savefig("gaussian_1000obs.pdf", bbox_inches='tight', transparent=True, pad_inches=0)

### ============================================================ ###
# 2.a Restricted range no MF
### ============================================================ ###
domain = (1.5,2.5)

## Generate some data
nb_init = 300
x_plot = np.linspace(*domain, 1000)[:, np.newaxis]
#x_plot = np.linspace(*(-10, 10), 1000)[:, np.newaxis]
x_init = np.linspace(*domain, nb_init)[:, np.newaxis] #
x_init = rdm.uniform(*domain, nb_init)[:, np.newaxis]
y_init = func(x_init)


## Define the Model used: a GP with an Exponential kernel + Binomial likelihood 
## + Laplace approximation
# specify the kernel k and the value of its initial hyperparameters
k = GPy.kern.Matern52(input_dim = 1, variance = 1., lengthscale = (domain[1]-domain[0])/20)
k.variance.constrain_bounded(1e-2, 5)
k.lengthscale.constrain_bounded(1e-2, 5)

# specify the inference method to use (Laplace approximation)
i_meth = GPy.inference.latent_function_inference.Laplace()

# specify the likelihood
lik = GPy.likelihoods.Binomial()

# Create the GP model
g = GPy.core.GP(X=x_init, Y=y_init, kernel=k, inference_method=i_meth, 
        likelihood=lik, Y_metadata = {'trials':np.ones_like(y_init) * NB_SHOTS})

# fit the hyperparameters
_ = g.optimize_restarts(num_restarts=5) 

visualize(proba, g, x_plot, x_init, y_init)
#plt.savefig("gaussian_1000obs.pdf", bbox_inches='tight', transparent=True, pad_inches=0)


### ============================================================ ###
# 2.b Use of a mean function
### ============================================================ ###
MF_VAL = invcdf(np.average(y_init))
mf = GPy.core.Mapping(input_dim=1, output_dim=1)
mf.f = lambda x: np.array([[MF_VAL] for xx in np.atleast_2d(x)])
mf.update_gradients = lambda a,b: 0
mf.gradients_X = lambda a,b: 0
# Create the GP model
gmf = GPy.core.GP(X=x_init, Y=y_init, kernel=k, inference_method=i_meth, 
        likelihood=lik, Y_metadata = {'trials':np.ones_like(y_init) * NB_SHOTS},
        mean_function=mf)

# fit the hyperparameters
_ = gmf.optimize_restarts(num_restarts=5) 

visualize(proba, gmf, x_plot, x_init, y_init)
#plt.savefig("gaussian_1000obs.pdf", bbox_inches='tight', transparent=True, pad_inches=0)


### ============================================================ ###
# 3.a Super restricted range no MF
### ============================================================ ###
domain = (1.75,1.95)

## Generate some data
nb_init = 100
x_plot = np.linspace(*domain, 1000)[:, np.newaxis]
#x_plot = np.linspace(*(-10, 10), 1000)[:, np.newaxis]
x_init = np.linspace(*domain, nb_init)[:, np.newaxis] #
x_init = rdm.uniform(*domain, nb_init)[:, np.newaxis]
y_init = func(x_init)


## Define the Model used: a GP with an Exponential kernel + Binomial likelihood 
## + Laplace approximation
# specify the kernel k and the value of its initial hyperparameters
k = GPy.kern.Matern52(input_dim = 1, variance = 1., lengthscale = (domain[1]-domain[0])/20)
k.variance.constrain_bounded(1e-2, 5)
k.lengthscale.constrain_bounded(1e-2, 5)

# specify the inference method to use (Laplace approximation)
i_meth = GPy.inference.latent_function_inference.Laplace()

# specify the likelihood
lik = GPy.likelihoods.Binomial()

# Create the GP model
g = GPy.core.GP(X=x_init, Y=y_init, kernel=k, inference_method=i_meth, 
        likelihood=lik, Y_metadata = {'trials':np.ones_like(y_init) * NB_SHOTS})

# fit the hyperparameters
_ = g.optimize_restarts(num_restarts=5) 

visualize(proba, g, x_plot, x_init, y_init/NB_SHOTS)
#plt.savefig("gaussian_1000obs.pdf", bbox_inches='tight', transparent=True, pad_inches=0)
print(g)

### ============================================================ ###
# 3.b rescale
### ============================================================ ###
## Generate some data
m  = -(domain[0] + domain[1])
alpha = (domain[1]-domain[0])/2
scale = lambda x: (x + m)/alpha
descale = lambda x: alpha * x - m
x_init2 = scale(x_init)


## Define the Model used: a GP with an Exponential kernel + Binomial likelihood 
## + Laplace approximation
# specify the kernel k and the value of its initial hyperparameters
k = GPy.kern.Matern52(input_dim = 1, variance = 1., lengthscale = (domain[1]-domain[0])/20)
k.variance.constrain_bounded(1e-2, 5)
k.lengthscale.constrain_bounded(1e-2, 5)

# specify the inference method to use (Laplace approximation)
i_meth = GPy.inference.latent_function_inference.Laplace()

# specify the likelihood
lik = GPy.likelihoods.Binomial()

# Create the GP model
grescale = GPy.core.GP(X=x_init2, Y=y_init, kernel=k, inference_method=i_meth, 
        likelihood=lik, Y_metadata = {'trials':np.ones_like(y_init) * NB_SHOTS})

# fit the hyperparameters
_ = grescale.optimize_restarts(num_restarts=5) 


probascale = lambda x: proba(descale(x))
x_plotscale = scale(x_plot)
visualize(probascale, grescale, x_plotscale, x_init2, y_init/NB_SHOTS)
#plt.savefig("gaussian_1000obs.pdf", bbox_inches='tight', transparent=True, pad_inches=0)
print(grescale)

### ============================================================ ###
# 3.b rescale + mf
### ============================================================ ###
MF_VAL = invcdf(np.average(y_init))
mf = GPy.core.Mapping(input_dim=1, output_dim=1)
mf.f = lambda x: np.array([[MF_VAL] for xx in np.atleast_2d(x)])
mf.update_gradients = lambda a,b: 0
mf.gradients_X = lambda a,b: 0
# Create the GP model
gmfrescale = GPy.core.GP(X=x_init2, Y=y_init, kernel=k, inference_method=i_meth, 
        likelihood=lik, Y_metadata = {'trials':np.ones_like(y_init) * NB_SHOTS},
        mean_function=mf)

# fit the hyperparameters
_ = gmfrescale.optimize_restarts(num_restarts=5) 

visualize(probascale, gmfrescale, x_plotscale, x_init2, y_init/NB_SHOTS)
print(gmfrescale)

