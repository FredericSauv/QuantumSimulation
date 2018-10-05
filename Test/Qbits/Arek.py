#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 19:58:08 2018

@author: fred
"""
from scipy.integrate import ode
import numpy as np
import matplotlib.pylab as plt

# -------------------------------- #
# Define Parameters of the system
# ---------------------------------#
m, g= 1,1
Lambda = 1/100
BigOmega = np.pi**(2/3)/(3**(1/3))
Omega = BigOmega
T = (2*np.pi)/Omega
href = np.pi**2/(2*(BigOmega)**2)
tmax = T
Delta = 1

# -------------------------------- #
# Define functions
# ---------------------------------#
def derivative(t, X, m = 1., g = 1., Lambda = 0.01, Omega = np.pi**(2/3)/(3**(1/3))):
    """ X = [x, p], X'(t) = [p(t)/m, -m * g * Lambda * cos(Omega * t) * (-1 if X[t]==0])"""
    if(X[0]<=0):
        X[1] = abs(X[1])
    dx = 1/m * X[1] 
    dy = (-m*g -Lambda * np.cos(Omega * t))
    return np.array([dx ,dy ])  

def evolve_ode(derivative, X0, t0, times, verbose= False, complex_type = False):
    """ wrap the ODE solver
    - derivative: func(t, X)
    
    """
    x0 = np.atleast_2d(X0) 
    shape0 = np.shape(x0)
    
    ## Set-up the solver
    solver = ode(derivative, jac=None)
    solver_args = {'nsteps': np.iinfo(np.int32).max, 'rtol':1e-9, 'atol':1e-9}
    solver.set_integrator('dop853', **solver_args)
    solver.set_initial_value(X0, t0)	
	
    ## Prepare container for the output
    if complex_type:
        v = np.empty(shape0+(len(times),),dtype=np.complex128)
    else:
        v = np.empty(shape0+(len(times),),dtype=np.float64)

    ## Run
    for i,t in enumerate(times):
        if t == t0:
            if verbose: print("evolved to time {0}, norm of state {1}".format(t,np.linalg.norm(solver.y)))
            v[...,i] = x0
            continue

        solver.integrate(t)
        if solver.successful():
            if verbose: print("evolved to time {0}, norm of state {1}".format(t,np.linalg.norm(solver.y)))
            v[...,i] = solver._y
        else:
            raise RuntimeError("failed to evolve to time {0}, nsteps might be too small".format(t))			
    return v


# -------------------------------- #
# Run
# ---------------------------------#
X0 = [href, 0.0]
t0 = 0.0
times = np.linspace(t0, T, 2)
test = evolve_ode(derivative, X0, t0, times)
print(test[0][0][-1]-href)



# -------------------------------- #
# Same results
# ---------------------------------#
h_0 = np.linspace(href - 2 * Delta, href + 20 * Delta, 500)
scan = np.array([(h_init, evolve_ode(derivative, [h_init, 0.0], t0, times)[0][0][-1] - href) for h_init in h_0])
plt.plot(scan[:,0], scan[:,1])






