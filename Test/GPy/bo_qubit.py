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

def measure(x, underlying_p = proba, nb_measures = 1):
    return rdm.binomial(nb_measures, underlying_p(x))/nb_measures

def squerror(p_pred, p_true):
    eps = np.squeeze(p_pred - p_true)
    return np.dot(eps, eps) /len(eps)

#def train_test(model, ):

### ============================================================ ###
# Data training / test
### ============================================================ ###
x_range = (0, np.pi)
nb_obs = 500
nb_meas = 10
nb_obs_nm = int(nb_obs/nb_meas)
x_test = np.linspace(*x_range)[:, np.newaxis]
p_test = proba(x_test)
x_train_onem = rdm.uniform(*x_range, nb_obs)[:, np.newaxis]
y_train_onem = measure(x_train_onem)
x_train_nm = rdm.uniform(*x_range, nb_obs_nm)[:, np.newaxis]
y_train_nm = measure(x_train_nm, nb_measures = nb_meas)

plt.plot(x_test, p_test, label = 'real proba')
plt.scatter(x_train_onem, y_train_onem, label='one shot')
plt.scatter(x_train_nm, y_train_nm, color='r', label=str(nb_meas) +' shots')
plt.legend()


### ============================================================ ###
# Regression task
# 1- Simple GP with heteroskedastic noise - onem
# 1b- Simple GP with heteroskedastic noise - nm
# 2- Learn warping - onem
# 2b- Learn warping - nm
# 3- As a classification task
### ============================================================ ###
#Same kernel for all
k_one = GPy.kern.Matern52(input_dim = 1, variance = 1., lengthscale = (x_range[1]-x_range[0])/4)
k_n = GPy.kern.Matern52(input_dim = 1, variance = 1., lengthscale = (x_range[1]-x_range[0])/4)
k_warp_one = GPy.kern.Matern52(input_dim = 1, variance = 1., lengthscale = (x_range[1]-x_range[0])/4)
k_warp_n = GPy.kern.Matern52(input_dim = 1, variance = 1., lengthscale = (x_range[1]-x_range[0])/4)
k_classi = GPy.kern.Matern52(input_dim = 1, variance = 1., lengthscale = (x_range[1]-x_range[0])/4)
k_classi_n = GPy.kern.Matern52(input_dim = 1, variance = 1., lengthscale = (x_range[1]-x_range[0])/4)
# ------------------------------------------ #
# 1
# ------------------------------------------ #
m_reg_one = GPy.models.GPRegression(X = x_train_onem, Y=y_train_onem, kernel=k_one)
m_reg_one.optimize_restarts(num_restarts = 10)
yp_reg_one, _ = m_reg_one.predict(x_test)
error_reg_one = squerror(yp_reg_one, p_test)
print(error_reg_one)
m_reg_one.plot()
plt.plot(x_test, p_test, 'r', label = 'real p')
plt.ylim([0.0,1.0])



m_reg_n = GPy.models.GPRegression(X = x_train_nm, Y=y_train_nm, kernel=k_n)
m_reg_n.optimize_restarts(num_restarts = 10)
yp_reg_n, _ = m_reg_n.predict(x_test)
error_reg_n = squerror(yp_reg_n, p_test)
print(error_reg_n)
m_reg_n.plot()
plt.plot(x_test, p_test, 'r', label = 'real p')
plt.ylim([0.0,1.0])


# 2 - With output warping


#2b with n measures
f_warp = GPy.util.warping_functions.TanhFunction(n_terms=3)
m_warp_n = GPy.models.warped_gp.WarpedGP(X=x_train_nm, Y=y_train_nm, kernel = k_warp_n, warping_function=f_warp)
m_warp_n['.*\.d'].constrain_fixed(1.0)
m_warp_n.optimize_restarts(num_restarts = 10)

m_warp_n.predict_in_warped_space = False
m_warp_n.plot(title="Warped GP - Latent space")
m_warp_n.predict_in_warped_space = True
m_warp_n.plot(title="Warped GP - Warped space")
plt.plot(x_test, p_test, 'r', label = 'real p')
plt.ylim([0.0,1.0])
plt.xlim(x_range)


m_warp_n.plot_warping()

yp_warp_n, _ = m_warp_n.predict(x_test)
error_warp_n = squerror(yp_warp_n, p_test)
print(error_warp_n)




# 3a Bernouilly
i_meth = GPy.inference.latent_function_inference.expectation_propagation.EP()
lik = GPy.likelihoods.Bernoulli()
m_classi = GPy.core.GP(X=x_train_onem, Y=y_train_onem, kernel=k_classi, 
                inference_method=i_meth, likelihood=lik)
_ = m_classi.optimize_restarts(num_restarts=10) #first runs EP and then optimizes the kernel parameters

yp_classi, _ = m_classi.predict(x_test)
test, _ = m_classi.predict_noiseless(x_test)
plt.plot(x_test, test)


error_classi = squerror(yp_classi, p_test)
print(error_classi)

m_classi.plot_f()
plt.plot(x_test, p_test, 'r', label = 'real p')
plt.ylim([0.0,1.0])



# 3b Binomial
i_meth_LP = GPy.inference.latent_function_inference.Laplace()
lik = GPy.likelihoods.Binomial()
Y_metadata={'trials':np.ones_like(y_train_nm) * nb_meas}
m_classi_n = GPy.core.GP(X=x_train_nm, Y=y_train_nm * nb_meas, kernel=k_classi_n, 
                inference_method=i_meth_LP, likelihood=lik, Y_metadata=Y_metadata)
_ = m_classi_n.optimize_restarts(num_restarts=10) #first runs EP and then optimizes the kernel parameters

yp_classi_n_tmp, _ = m_classi_n.predict_noiseless(x_test)
yp_classi_n = m_classi_n.likelihood.gp_link.transf(yp_classi_n_tmp) 

plt.plot(x_test, yp_classi_n)
plt.plot(x_test, p_test, 'r', label = 'real p')
plt.ylim([0.0,1.0])
error_classi_n = squerror(yp_classi_n, p_test)
print(error_classi_n)






### ============================================================ ###
# Testingg
### ============================================================ ###
X = (2 * np.pi) * np.random.random(151) - np.pi
Y = np.sin(X) + np.random.normal(0,0.2,151)
Y = np.array([np.power(abs(y),float(1)/3) * (1,-1)[y<0] for y in Y])
X = X[:, None]
Y = Y[:, None]

warp_k = GPy.kern.RBF(1)
warp_f = GPy.util.warping_functions.TanhFunction(n_terms=2)
warp_m = GPy.models.WarpedGP(X, Y, kernel=warp_k, warping_function=warp_f)
warp_m['.*\.d'].constrain_fixed(1.0)
m = GPy.models.GPRegression(X, Y)
m.optimize_restarts(parallel=False, robust=True, num_restarts=5, max_iters=10)
warp_m.optimize_restarts(parallel=False, robust=True, num_restarts=5, max_iters=10)
#m.optimize(max_iters=max_iters)
#warp_m.optimize(max_iters=max_iters)

print(warp_m)
print(warp_m['.*warp.*'])

warp_m.predict_in_warped_space = False
warp_m.plot(title="Warped GP - Latent space")
warp_m.predict_in_warped_space = True
warp_m.plot(title="Warped GP - Warped space")
m.plot(title="Standard GP")
warp_m.plot_warping()
pb.show()
