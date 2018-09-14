import functools as ft
import numpy as np
import numpy.random as rdm
import matplotlib.pylab as plt
import GPy
import GPyOpt


#-----------------------------------------------------------------------------#
# UNDERLYING DYNAMICS
# phi = cos(x) |0> + sin(x) |1>
# Measurement(n, phi(x)) = B(sin^2(x), n) /n 
#-----------------------------------------------------------------------------#
def p(x):
    return np.sin(x)**2 

def m(x, n=100):
    return np.random.binomial(n,p(x))/n
    
def square_error(x, y):
    eps = np.squeeze(x) - np.squeeze(y)
    return np.dot(eps, eps) / len(x)

#def transfo(x, strtch = 15):
#    coeff = 1/np.tanh(strtch*0.5)
#    return 0.5 + coeff/2 * np.tanh(strtch*(x-0.5))

def t_one(x):
    return 1/2+np.sin(np.pi*x- np.pi/2)/2

def t_one_inv(x):
    return 1/2+np.arcsin(2*x-1)/np.pi

def var_rough(p):
    return np.sqrt(p*(1-p))

x_to_predict = np.arange(0, np.pi, 0.01)[:, np.newaxis]
y_true  = p(x_to_predict)

p_test = np.arange(0,1.01,0.01)
plt.plot(p_test, var_rough(p_test))
plt.plot(p_test, 1/2*np.pi)
plt.plot(p_test, t_one_inv(t_one(p_test)))


#-----------------------------------------------------------------------------#
# REGRESSION TASK
#-----------------------------------------------------------------------------#
# Data
range_x = [0, np.pi]
nb_meas = 5
nb_rdm_obs = 20
x_obs = np.random.uniform(range_x[0], range_x[1], size=nb_rdm_obs)[:, np.newaxis]
y_obs = m(x_obs, nb_meas)

plt.plot(x_to_predict, y_true)
plt.scatter(x_obs, y_obs, color = 'red')

# model
ker = GPy.kern.Matern52(input_dim = 1) 
ker_with_noise =GPy.kern.Matern52(input_dim = 1) + GPy.kern.White(input_dim = 1)
reg = GPy.models.GPRegression(x_obs, y_obs, ker)
reg.optimize()
#reg.plot()
y_predict, v_predict = reg.predict(x_to_predict)
print(square_error(y_predict, y_true))


#-----------------------------------------------------------------------------#
# FIXED TRANSFO
#-----------------------------------------------------------------------------#
y_obs_transfo = t_one_inv(y_obs)
ker = GPy.kern.Matern52(input_dim = 1) 
reg2 = GPy.models.GPRegression(x_obs, y_obs_transfo, ker)
reg2.optimize()
#reg2.plot()
y2_predict, v2_predict = reg2.predict(x_to_predict)
print(square_error(t_one(y2_predict), y_true))


#-----------------------------------------------------------------------------#
# WARPING
#-----------------------------------------------------------------------------# 
# model
ker = GPy.kern.Matern52(input_dim = 1) 
reg3 = GPy.models.warped_gp.WarpedGP(x_obs, y_obs, ker)
reg3.optimize()
#reg.plot()
y3_predict, v3_predict = reg3.predict(x_to_predict)
print(square_error(y3_predict, y_true))


    X = np.random.randn(100, 1)
    Y = np.sin(X) + np.random.randn(100, 1)*0.05

    m = WarpedGP(X, Y)

ker = GPy.kern.Matern52(input_dim = 1) 
ker_with_noise =GPy.kern.Matern52(input_dim = 1) + GPy.kern.White(input_dim = 1)
m = GPy.models.GPRegression(x_obs, y_obs, ker)
m_with_noise= GPy.models.GPRegression(x_obs, y_obs, ker_with_noise)

m.optimize(messages=True)
fig = m.plot()
fig.plot(x, y_perfect, '--',color = 'r', label = 'f')
plt.xlim([0, np.pi])


# Learn with heteroscedastic noise
nb_obs = 10
noise = 0.5
sd_weird_noise = ft.partial(sd_f, sd = noise)
f_weird_noise = ft.partial(f_to_learn, mean_f = mean_f, sd_f = sd_weird_noise)
y_obs = f_weird_noise(x_obs)
m = GPy.models.GPRegression(x_obs, y_obs, ker)
m.optimize(messages=True)
fig = m.plot()
fig.plot(x, y_perfect, '--',color = 'r', label = 'f')
plt.xlim([0, np.pi])


# With warping

