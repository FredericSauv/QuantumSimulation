#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 13:46:15 2018

@author: frederic
"""
import numpy as np
import numpy.random as rdm
import matplotlib.pylab as plt
import GPy
import scipy.stats as stats
import scipy.special as spe
from matplotlib.ticker import MultipleLocator

### ============================================================ ###
# MODEL: restricted qubits with projective measurement
# |phi(x)> = sin(x) |1> + cos(x) |0>
### ============================================================ ###
def proba(x, noise=0):
    #generate underlying proba p(|1>)
    if(noise>0):
        x_noise = x + rdm.normal(0, noise, size = x.shape)
    else:
        x_noise = x
    return np.square(np.sin(x_noise))

def proba2(x, noise=0):
    #generate underlying proba p(|1>)
    if(noise>0):
        x_noise = x + rdm.normal(0, noise, size = x.shape)
    else:
        x_noise = +np.sin(3*(x+0.3))/2 + (x+0.3)/1.5
    return np.square(np.sin(x_noise))

def counts(x, underlying_p = proba, nb_measures = 1):
    return rdm.binomial(nb_measures, underlying_p(x))

def measure(x, underlying_p = proba, nb_measures = 1):
    return rdm.binomial(nb_measures, underlying_p(x))/nb_measures

def squerror(p_pred, p_true):
    eps = np.squeeze(p_pred - p_true)
    return np.dot(eps, eps) /len(eps)

def predict_p(model, X):
    mean, var = model._raw_predict(X, full_cov=False, kern=None)
    likelihood = model.likelihood
    Nf_samp = 10000
    s = np.random.randn(mean.shape[0], Nf_samp) * np.sqrt(var) + mean
    #ss_y = self.samples(s, Y_metadata, samples=Ny_samp)
    p = likelihood.gp_link.transf(s)
    p_min = np.min(p)
    p_max = np.max(p)
    mean = np.median(p, axis = 1)
    std = np.std(p, axis = 1)
    q = np.quantile(p, [0.025, 0.975], axis=1)
    y = np.linspace(0, 1)
    density = np.diff(np.vstack([np.array([[np.sum(pp < yy) for pp in p] for yy in y]), np.ones(len(p))*Nf_samp]), axis=0)/Nf_samp
    
    return mean, q[0], q[1], density, (p_min, p_max), std

def invcdf(y, alpha=1):
    return  np.sqrt(2) * spe.erfinv(2*y-1) / alpha 

def changedistrib(y, mu, var, alpha=1):
    inv = invcdf(y,alpha)
    res = stats.norm.pdf(inv, loc=mu, scale=var) / (alpha * stats.norm.pdf( alpha * inv))
    res[np.isnan(res)]=0
    return  res
    
def predict_p_v2(model, X, alpha=1):
    mean, var = model._raw_predict(X, full_cov=False, kern=None)
    #likelihood = model.likelihood
    #Nf_samp = 10000
    #s = np.random.randn(mean.shape[0], Nf_samp) * np.sqrt(var) + mean
    #ss_y = self.samples(s, Y_metadata, samples=Ny_samp)
    #p = likelihood.gp_link.transf(s)
    N_discr = 5000
    y = np.linspace(0, 1,N_discr)
    density = np.array([changedistrib(y, m, np.sqrt(v), alpha) for m, v in zip(mean, var)])
    p_min, p_max = 0, 1
    m_mean = np.array([np.sum(d * y)/len(y) for d in density])
    densitycum = np.cumsum(density, axis=1)/N_discr
    try:
      m_median = np.array([y[np.argwhere(dc>=0.5)[0,0]] for dc in densitycum])
    except:
      print("pb median")
      m_median = m_mean
    try:
      q = np.array([(y[np.argwhere(dc>=0.025)[0,0]], y[np.argwhere(dc<=0.975)[-1,0]]) for dc in densitycum])
    except:
      q = np.stack([np.zeros(len(y)), np.zeros(len(y))], axis=1)
    std = np.array([np.sqrt(np.sum(np.square(y-m) * d)/N_discr) for d, m in zip(density, m_mean)])
    return m_median, q[:,0], q[:,1], np.transpose(density), (p_min, p_max), std


def next_ucb_bounds(model, loc, w=4):
    a, b, c, d, d_range, s = predict_p(model, loc)
    acq = (a + w * (c-b))
    x_next = loc[np.argmax(acq)]
    return x_next

def next_ucb_std(model, loc, w=4):
    a, b, c, d, d_range, s = predict_p(model, loc)
    acq = (a + w * s)
    x_next = loc[np.argmax(acq)]
    return x_next

def next_ucb_bounds_v2(model, loc, w=4, alpha=1):
    a, b, c, d, d_range, s = predict_p_v2(model, loc, alpha)
    acq = (a + w * (c-b))
    x_next = loc[np.argmax(acq)]
    return x_next

def next_ucb_std_v2(model, loc, w=4, alpha=1):
    a, b, c, d, d_range, s = predict_p_v2(model, loc, alpha)
    acq = (a + w * s)
    x_next = loc[np.argmax(acq)]
    return x_next

### ============================================================ ###
# Meta stuff
### ============================================================ ###
save=False
weight=6
col_custom = (0.1, 0.2, 0.5)
weight = 4
font_size=28
points_size = 50



### ============================================================ ###
# Data training / test
### ============================================================ ###

def runBO(nb_iter, nb_meas=1, binmode = True, nb_init = None):
  seed = np.random.randint(1,1000000000)
  np.random.seed(seed)
  if nb_init is None:
    nb_init = int(max(30 / nb_meas, 5))  
  x_range = (0, 4)
  
  if binmode:
    func = lambda x: counts(x, underlying_p = proba2, nb_measures=nb_meas)
    k = GPy.kern.Matern52(input_dim = 1, variance = 1., lengthscale = (x_range[1]-x_range[0])/25)
    i_meth = GPy.inference.latent_function_inference.Laplace()
    lik = GPy.likelihoods.Binomial()

  else:
    func = lambda x: measure(x, underlying_p = proba2, nb_measures=nb_meas)
    k = GPy.kern.Matern52(input_dim = 1, variance = 1., lengthscale = (x_range[1]-x_range[0])/25)
    i_meth = None
    lik = GPy.likelihoods.Gaussian()
    
  
  x_test = np.linspace(*x_range, 1000)[:, np.newaxis]
  x_init = rdm.uniform(*x_range, nb_init)[:, np.newaxis]
  y_init = func(x_init)
    
  m = GPy.core.GP(X=x_init, Y=y_init, kernel=k, 
                  inference_method=i_meth, likelihood=lik, Y_metadata = {'trials':np.ones_like(y_init) * nb_meas})
  _ = m.optimize_restarts(num_restarts=5) 

  for n in range(nb_iter):
      x_next = np.atleast_1d(next_ucb_std_v2(m, x_test, weight))
      y_next = func(x_next)
      X_new = np.vstack([m.X, x_next])
      Y_new = np.vstack([m.Y, y_next])
      m.Y_metadata = {'trials':np.ones_like(Y_new)*nb_meas}
      m.set_XY(X_new, Y_new)    
      _ = m.optimize_restarts(num_restarts=2, verbose=False)
  
  x_best = next_ucb_std_v2(m, m.X, 0)
  pres = proba2(x_best)
  print(pres)
  return pres
#ax1.text(-0.4, 1.1, '(c)', fontsize=font_size)
#ax2.set_ylabel(r'$ a.u.$', fontsize=16)

res0 = [runBO(70, 1, True) for _ in range(30)]
res0g = [runBO(70, 1, False) for _ in range(30)]
res1 = [runBO(5, 10, True) for _ in range(10)]
res2 = [runBO(10, 10, True) for _ in range(10)]
res2b = [runBO(15, 10, True) for _ in range(10)]
res2bg = [runBO(15, 10, False) for _ in range(10)]
res3 = [runBO(5, 25, True) for _ in range(10)]
res4 = [runBO(10, 25, True) for _ in range(10)]
res5 = [runBO(4, 25, True) for _ in range(10)]
res6 = [runBO(3, 50, True) for _ in range(10)]


res1g = [runBO(5, 10, False) for _ in range(10)]
res2g = [runBO(10, 10, False) for _ in range(10)]
res3g = [runBO(5, 25, False) for _ in range(10)]
res4g = [runBO(10, 25, False) for _ in range(10)]
res5g = [runBO(4, 25, False) for _ in range(10)]
res6g = [runBO(3, 50, True) for _ in range(10)]

print(np.average(res0), np.median(res0)) #0.9565148738451951 0.9650889741055936 XXXXX 100/100
print(np.average(res0g), np.median(res0g)) #0.9533661775861786 0.9847030293485426

print(np.average(res1), np.median(res1)) # 0.9236261587605362 0.9076737214163408
print(np.average(res1g), np.median(res1g)) #0.7682682521692981 0.7702624029916378

print(np.average(res2), np.median(res2)) #0.9160244292712555 0.9223958860182464
print(np.average(res2g), np.median(res2g)) #0.884244012375279 0.9130454366818107
print(np.average(res2b), np.median(res2b)) #0.9160244292712555 0.9223958860182464 XXXX 25/250
print(np.average(res2bg), np.median(res2bg)) #0.884244012375279 0.9130454366818107

print(np.average(res3), np.median(res3)) #0.8949466954030225 0.9084024027853246
print(np.average(res3g), np.median(res3g)) #0.8722236443464529 0.9186405008923909

print(np.average(res4), np.median(res4)) #0.97087376564578 0.9954097914626646 XXXXX 15/325
print(np.average(res4g), np.median(res4g))#0.9166108465600857 0.9258958272067428

print(np.average(res5), np.median(res5)) #0.912279433451533 0.8964870308068571
print(np.average(res5g), np.median(res5g)) #0.8041394485670459 0.8053235684113254

print(np.average(res5), np.median(res5)) #0.912279433451533 0.8964870308068571
print(np.average(res5g), np.median(res5g)) #0.8041394485670459 0.8053235684113254

print(np.average(res6), np.median(res5)) #0.922978462334252 0.8964870308068571
print(np.average(res6g), np.median(res5g)) #0.9023375409341667 0.8053235684113254