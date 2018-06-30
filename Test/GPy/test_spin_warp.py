import functools as ft
import numpy as np
import numpy.random as rdm
import matplotlib.pylab as plt
import GPy
import GPyOpt

def mean_f(x):
    return np.sin(x)**2 

def sd_f_constant(x, sd = 0.1):
    return sd

def sd_f(x, sd = 0.1):
    return sd * np.sqrt(np.sin(x * 2)**2)

def f_to_learn(x, mean_f, sd_f = lambda x: 0):
    res = np.random.normal(mean_f(x), sd_f(x))
    return res
    
## True function
x = np.arange(-np.pi, 2 * np.pi, 0.01)
y_perfect  = f_to_learn(x, mean_f)
plt.scatter(x, y_perfect)


## Regression    
nb_obs = 10

# Learn with no noise
f_no_noise = ft.partial(f_to_learn, mean_f = mean_f)
x_obs = rdm.uniform(0, np.pi, (nb_obs,1))
y_obs = f_no_noise(x_obs)
 

ker = GPy.kern.Matern52(input_dim = 1) 
ker_with_noise =GPy.kern.Matern52(input_dim = 1) + GPy.kern.White(input_dim = 1)
m = GPy.models.GPRegression(x_obs, y_obs, ker)

fig = m.plot()
fig.plot(x, y_perfect, '--',color = 'r')
plt.xlim([0,np.pi])

m.optimize(messages=True)
fig = m.plot()
fig.plot(x, y_perfect, '--',color = 'r', label = 'f')
plt.xlim([-np.pi, 2 * np.pi])


# Learn with constant noise
nb_obs = 10
noise = 0.3
x_obs = rdm.uniform(0, np.pi, (nb_obs,1))
sd_constant_noise = ft.partial(sd_f_constant, sd = noise)
f_constant_noise = ft.partial(f_to_learn, mean_f = mean_f, sd_f = sd_constant_noise)

y_obs = f_constant_noise(x_obs)
 

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

