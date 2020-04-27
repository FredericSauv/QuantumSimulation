"""
Created on Jan 8 19:58:08 2019
@author: FS
"""
from scipy.integrate import ode
import numpy as np
import matplotlib.pylab as plt
import functools
from matplotlib import rc
rc('text', usetex=True)
from scipy import optimize as optim
#import pdb



# TODO:
# a. compute effective Hamiltonian under polychromatic drive
# b. analytic formulas for PHO with bichromatic drive
# c. Verif with more oscillators the heating rate
# d. optim 6/6 really closer IN PROGRESS
# e. optim 4/4 really closer IN PROGRESS
# f. Analytic formulas A(t) cos(w * t) 
# g. Can it be deal with Adiabatic treatment of Heff (cf. Viktor)
# h. be able to simulate GPE and see
# i. 
class param_ho:
    """ Parametric harmonic oscillator
    d^2/dt^2 x(t) + w0^2 * (1 + f(t)) x(t) = 0
    
    Parameters
    ---------
    m: mass of the system
    w0: effective mass of the system
    drive_f: array containing the frequencies of the driving
    drive_a: array containing the amplitudes of the driving
    drive_p: array containing the phases of the driving
    x_0: initial position
    dx_0: initial velocity
        
    Notations
    ---------
    X: state of the system [x, dx]
    x: position of the system
    dx: dx/dt
    
    In free space the classical dynamics is given by:
        d^2/dt^2 x(t) + w0^2 * (1 + f(t)) x(t) = 0
        
    f(t) = sum_i drive_a[i] * cos(drive_f[i] * t + drive_p[i])     
        
    """
    
    def __init__(self, drive_f, drive_a, drive_p, m=1, w0 = 1): 
        """ Initialize properties of the model
        """
        self.drive_f = np.atleast_1d(drive_f)
        self.drive_a = np.atleast_1d(drive_a)
        self.drive_p = np.atleast_1d(drive_p)
        self.w0 = w0
        self.m = m

    def get_kin_en(self, X):
        """ Get the kinetic energy of the system at time t
        """        
        return 0.5 * self.m * np.square(X[1])
        
    def get_pot_en(self, X, t):
        """ Get the potential energy of the system at time t
        """        
        return 0.5 * self.w0 * (1 + self.get_drive(t)) * (np.square(X[0]))
           
        
    def get_en(self, X, t):
        """ Get the total energy of the system at time t
        """        
        return self.get_kin_en(X) + self.get_pot_en(X, t)
        
    def get_drive(self, t):
        """ Drive at time t
        """
        return np.squeeze(np.sum(self.drive_a * np.cos(np.outer(np.atleast_1d(t),self.drive_f) + self.drive_p), 1))
        
        
    def evol_to_t(self, t, x0 = 0.05, dx0 = 0, energy = False):
        """ return [x(t), dx(t), en(t)] x(t) = [x0, ..., xT]
        shape = (2 x len(t))
        """
        evol = param_ho.evolve_ode(self.derivative, [x0, dx0], t0=0, times=np.atleast_1d(t))
        if energy:
            en = self.get_en(evol, np.atleast_1d(t))
            evol = np.vstack((evol, en))
        return evol
    
    def derivative(self, t, X):
        """ X = [x, dx], X'(t) = [dx, ddx] = [dx, -w0^2*(1+f(t)) x(t) ]"""
        ddx = - self.w0 ** 2 * (1 + self.get_drive(t)) * X[0]
        return np.array([X[1], ddx])  

    @staticmethod
    def evolve_ode(derivative, X0, t0, times, verbose= False, complex_type = False):
        """ wrap the ODE solver
        X state of the system (i.e. here [x, dx])
        Parameters
        ----------
            + derivative: func(t, X) -> X'(t)
            + X0: an array with thdrive_fe initial conditon
            + t0: starting time
            + times : List of times for which we want state o the system
            + verbose: print some info about the solver
            + complex_type: allow for complex values
        
        Output
        ------
        State of the system over the requested times shape = (size(X), time)
        i.e. transpose([X(times[0]), ..., X(times[-1])])
        
        """
        ## Set-up the solver
        solver = ode(derivative, jac=None)
        solver_args = {'nsteps': np.iinfo(np.int32).max, 'rtol':1e-9, 'atol':1e-9}
        solver.set_integrator('dop853', **solver_args)
        solver.set_initial_value(X0, t0)	
        	
        ## Prepare container for the output
        if complex_type:
            v = np.empty((len(X0),len(times)), dtype=np.complex128)
        else:
            v = np.empty((len(X0),len(times)), dtype=np.float64)
        
        ## Run
        for i,t in enumerate(times):
            if t == t0:
                if verbose: print("evolved to time {0}, norm of state {1}".format(t,np.linalg.norm(solver.y)))
                v[...,i] = X0
                continue
        
            solver.integrate(t)
            if solver.successful():
                if verbose: print("evolved to time {0}, norm of state {1}".format(t,np.linalg.norm(solver.y)))
                v[...,i] = solver._y
            else:
                raise RuntimeError("failed to evolve to time {0}, nsteps might be too small".format(t))			
        return v

def rough_exp(data, time, strobo_ind):
    exp_rough = np.log(data[strobo_ind][-1] / data[strobo_ind][0]) / time[strobo_ind][-1]
    print(exp_rough)
    return exp_rough

#Optimization
def costfunction_nH(params, model, time, x0=0.05, dx0=0):
    n = int(len(params)/2)
    model.drive_a[1:1+n] = params[0:n]
    model.drive_p[1:1+n] = params[n:2*n]
    ev = model.evol_to_t(t=time, x0=x0, dx0=dx0, energy = True)
    return ev[2]

def costfunction_multi_nHh(params, list_pho, time, x0=0.05, dx0=0):
    en_list = [costfunction_nH(params, pho, time, x0, dx0) for pho in list_pho]
    en_total = np.sum(en_list, 0)
    cost = np.max(en_total)
    print(cost/1000000)
    return cost

if False:
    w_int = 10
    t_fin = 20.2
    t_strobo = 2 * np.pi / w_int 
    nt_period = 50
    time = np.arange(0, t_fin, t_strobo / nt_period)
    strobo_ind = np.arange(int(nt_period/6), len(time), nt_period)

    
    # ------------- #
    # STUDY 1
    # Dynamic with a monochromatic drive
    # ------------- #
    drive_f = [2*w_int]
    drive_a = [0.2]
    drive_p = [0]
    pho = param_ho(drive_f, drive_a, drive_p, m=1, w0 = w_int)
    evol = pho.evol_to_t(t=time, x0=0.05, dx0 = 0, energy = True)
    toplot = evol[2, :]
    
    #Quickly fit an exponent
    exp_rough = rough_exp(toplot, time, strobo_ind)    
    # look at the dynamic
    plt.plot(time, toplot)
    plt.scatter(time[strobo_ind], toplot[strobo_ind])
    plt.plot(time, np.exp(exp_rough * time) * toplot[strobo_ind][0])
            
    
    # ------------- #
    # STUDY 2
    # Play with one pho with 3 harmonics
    # ------------- #
    drive_f = [20, 19.5, 20.5]
    drive_a = [0.2, 0.25, 0.25]
    drive_p = [0, -0.65, 0.65]
    pho_bi = param_ho(drive_f, drive_a, drive_p, m=1, w0 = w_int)
    evol = pho_bi.evol_to_t(t=time, x0=0.05, dx0 = 0, energy = True)
    toplot = evol[2]
    print(np.max(toplot)/1000000)
    plt.plot(time, toplot)
    plt.scatter(time[strobo_ind], toplot[strobo_ind])


    # ------------- #
    # STUDY 3
    # Optim of one pho drive on resonance and 3 harmonics
    # Doesn't work when run on longer periods (i.e. T = 40)
    # 
    # ------------- #
    #Model initial
    drive_f = [20, 19.5, 20.5]
    drive_a = [0.2, 0., 0.]
    drive_p = [0., 0., 0.]
    pho_ref = param_ho(drive_f, drive_a, drive_p, m=1, w0 = w_int)
    evol_ref = pho_ref.evol_to_t(t=time, x0=0.05, dx0 = 0, energy = True)
    cf_ref = np.max(evol_ref[2])
    
    #Optimization
    def costfunction_3h(params, model, time, x0=0.05, dx0=0):
        model.drive_a[1:3] = params[0:2]
        model.drive_p[1:3] = params[2:4]
        ev = model.evol_to_t(t=time, x0=x0, dx0=dx0, energy = True)
        cost = np.max(ev[2])
        print(cost/1000000)
        return cost

    bounds = [(-0.5, 0.5), (-0.5, 0.5), (0, 2 * np.pi),(0, 2*np.pi)]
    f = functools.partial(costfunction_3h, model=pho_ref,time=time)
    de_optim = optim.differential_evolution(f, bounds)

    #Optimized
    drive_f = [20, 19.5, 20.5]
    drive_a = [0.2, 3.38324186e-03, -2.98121466e-01]
    drive_p = [0., 3.93031947e+00, 4.64560877]
    pho_optim = param_ho(drive_f, drive_a, drive_p, m=1, w0 = w_int)
    evol_optim = pho_optim.evol_to_t(t=time, x0=-0.55, dx0 = 0, energy = True)
    cf_optim = np.max(evol_optim[2])
    print(cf_optim)
    
    toplot_optim = evol_optim[2]
    plt.plot(time, toplot_optim)
    plt.scatter(time[strobo_ind], toplot_optim[strobo_ind])
    
    toplot_ref = evol_ref[2]
    plt.plot(time, toplot_ref)
    plt.scatter(time[strobo_ind], toplot_ref[strobo_ind])
    plt.xlim([0,10.])
    plt.ylim([0,1000])
    
    
    # New questiojns
    # Plot dynamically the drive
    drive_optim = pho_optim.get_drive(time)
    drive_ref = pho_ref.get_drive(time)
    plt.plot(time, drive_ref)
    plt.plot(time, drive_optim)

    
    # Try with another init
    
    
    
    # ------------- #
    # STUDY 4
    # Optim 3 phos and 3 harmonics
    # ------------- #
    
    #Model initial
    drive_f = [20, 19.5, 20.5]
    drive_a = [0.2, 0., 0.]
    drive_p = [0., 0., 0.]
    phoS_ref = [param_ho(drive_f, drive_a, drive_p, m=1, w0 = w/2) for w in drive_f]
    evolS_ref = [p.evol_to_t(t=time, x0 = 0.05, dx0 = 0, energy=True) for p in phoS_ref]
    
    en_main = evolS_ref[0][2]
    en_total = np.sum([e[2] for e in evolS_ref], 0)
    plt.plot(time, en_main)
    plt.plot(time, en_total)    
    cfS_ref = np.max(en_total)
    print(cfS_ref)
    
    


    bounds = [(-0.5, 0.5), (-0.5, 0.5), (0, 2 * np.pi),(0, 2*np.pi)]
    f_multi = functools.partial(costfunction_multi_nHh, list_pho=phoS_ref,time=time)
    de_optim = optim.differential_evolution(f_multi, bounds)

    #Optimized 2 extra harmonics
    drive_f = [20, 19.5, 20.5]
    drive_a = [0.2,  0.14569457, -0.15768501]
    drive_p = [0., 3.82713268,  4.80639031]
    phoS_optim = [param_ho(drive_f, drive_a, drive_p, m=1, w0 = w/2) for w in drive_f]
    evol_optim = [p.evol_to_t(t=time, x0=0.05, dx0 = 0, energy = True) for p in phoS_optim]

    for er, eo, d in zip(evolS_ref, evol_optim, drive_f):
        plt.figure()
        plt.plot(time, er[2]/1000000)
        plt.plot(time, eo[2]/1000000)
        plt.title(str(d))
        
    plt.figure()
    plt.bar(drive_f, drive_a, width=0.1)


    # ------------- #
    # STUDY 5
    # Optim 5 phos and 5 harmonics
    # ------------- #
    
    #Model initial
    drive_f = [20, 19.5, 20.5, 19, 21]
    drive_a = [0.2, 0., 0., 0., 0.]
    drive_p = [0., 0., 0., 0., 0.]
    phoS5_ref = [param_ho(drive_f, drive_a, drive_p, m=1, w0 = w/2) for w in drive_f]
    evolS5_ref = [p.evol_to_t(t=time, x0 = 0.05, dx0 = 0, energy=True) for p in phoS5_ref]
    
    en_main5 = evolS5_ref[0][2]
    en_total5 = np.sum([e[2] for e in evolS5_ref], 0)
    plt.plot(time, en_main5)
    plt.plot(time, en_total5)    
    cfS5_ref = np.max(en_total5)
    print(cfS5_ref)
    
    bounds = [(-0.5, 0.5), (-0.5, 0.5),(-0.5, 0.5), (-0.5, 0.5), (0, 2 * np.pi),(0, 2*np.pi),(0, 2 * np.pi),(0, 2*np.pi)]
    f_multi5 = functools.partial(costfunction_multi_nHh, list_pho=phoS5_ref,time=time)
    de_optim = optim.differential_evolution(f_multi5, bounds)


    #Optimized 2 extra harmonics
    drive_a = [0.2,  -0.11739779,  0.14380509, -0.11182537,  0.11389427]
    drive_p = [0., 1.42137842, 1.11651784,  0.56955107,  2.08057063]
    phoS_optim5 = [param_ho(drive_f, drive_a, drive_p, m=1, w0 = w/2) for w in drive_f]
    evol_optim5 = [p.evol_to_t(t=time, x0=0.05, dx0 = 0, energy = True) for p in phoS_optim5]

    for er, eo, d in zip(evolS5_ref, evol_optim5, drive_f):
        plt.figure()
        plt.plot(time, er[2]/1000000)
        plt.plot(time, eo[2]/1000000)
        plt.title(str(d))
        
    plt.figure()
    plt.bar(drive_f, drive_a, width=0.1)
    
    
    drive_optim = phoS_optim5[0].get_drive(time)
    drive_ref = pho_ref.get_drive(time)
    plt.plot(time, drive_ref)
    plt.plot(time, drive_optim)
#    #Optimized
#    drive_f = [20, 19.5, 20.5]
#    drive_a = [0.2, 3.38324186e-03, -2.98121466e-01]
#    drive_p = [0., 3.93031947e+00, 4.64560877]
#    pho_optim = param_ho(drive_f, drive_a, drive_p, m=1, w0 = w_int)
#    evol_optim = pho_optim.evol_to_t(t=time, x0=0.25, dx0 = 0, energy = True)
#    cf_optim = np.max(evol_optim[2])
#    print(cf_optim)
#    
#    toplot_optim = evol_optim[2]
#    plt.plot(time, toplot_optim)
#    plt.scatter(time[strobo_ind], toplot_optim[strobo_ind])
#    
#    toplot_ref = evol_ref[2]
#    plt.plot(time, toplot_ref)
#    plt.scatter(time[strobo_ind], toplot_ref[strobo_ind])
#    plt.xlim([0,10.])
#    plt.ylim([0,1000])
#    
#    
#    # New questiojns
#    # Plot dynamically the drive
#    drive_optim = pho_optim.get_drive(time)
#    drive_ref = pho_ref.get_drive(time)
#    plt.plot(time, drive_ref)
#    plt.plot(time, drive_optim)
#
#    
#    # Try with another init
#    
#    
#    
##    
#
#    
##    


    # ------------- #
    # STUDY 9
    # Optim 5 phos and 5 harmonics
    # ------------- #
    
    #Model initial
    drive_f = [20, 19.75, 20.25, 19.5, 20.5]
    drive_a = [0.2, 0., 0., 0., 0.]
    drive_p = [0., 0., 0., 0., 0.]
    phoS5_ref = [param_ho(drive_f, drive_a, drive_p, m=1, w0 = w/2) for w in drive_f]
    evolS5_ref = [p.evol_to_t(t=time, x0 = 0.05, dx0 = 0, energy=True) for p in phoS5_ref]
    
    en_main5 = evolS5_ref[0][2]
    en_total5 = np.sum([e[2] for e in evolS5_ref], 0)
    plt.plot(time, en_main5)
    plt.plot(time, en_total5)    
    cfS5_ref = np.max(en_total5)
    print(cfS5_ref)
    0.71043255
    bounds = [(-0.5, 0.5), (-0.5, 0.5),(-0.5, 0.5), (-0.5, 0.5), (0, 2 * np.pi),(0, 2*np.pi),(0, 2 * np.pi),(0, 2*np.pi)]
    f_multi5 = functools.partial(costfunction_multi_nHh, list_pho=phoS5_ref,time=time)
    de_optim5 = optim.differential_evolution(f_multi5, bounds)


    #Optimized 2 extra harmonics
    drive_a = [0.2,  -0.1556261 ,  0.15473844,  0.11013981, -0.10598907]
    drive_p = [0., 2.61750446, 0.71043255,  4.91648615,  4.29076598]
    phoS_optim5 = [param_ho(drive_f, drive_a, drive_p, m=1, w0 = w/2) for w in drive_f]
    evol_optim5 = [p.evol_to_t(t=time, x0=0.05, dx0 = 0, energy = True) for p in phoS_optim5]

    en_main5 = evolS5_ref[0][2]
    en_total5 = np.sum([e[2] for e in evolS5_ref], 0)
    plt.plot(time, en_main5)
    plt.plot(time, en_total5)    
    cfS5_ref = np.max(en_total5)
    print(cfS5_ref)

    for er, eo, d in zip(evolS5_ref, evol_optim5, drive_f):
        plt.figure()
        plt.plot(time, er[2]/1000000)
        plt.plot(time, eo[2]/1000000)
        plt.title(str(d))
        plt.ylim([0, 0.001])
        
    plt.figure()
    plt.bar(drive_f, drive_a, width=0.1)
    
    drive_optim = phoS_optim5[0].get_drive(time)
    drive_ref = phoS5_ref[0].get_drive(time)
    plt.figure()
    plt.plot(time, drive_ref)
    plt.plot(time, drive_optim)
    
    # ------------- #
    # STUDY 10
    # Optim 5 phos and 5 harmonics
    # ------------- #
    
    #Model initial
    drive_f = [20, 19.95, 20.05, 19.9, 20.1]
    drive_a = [0.2, 0., 0., 0., 0.]
    drive_p = [0., 0., 0., 0., 0.]
    phoS5_ref = [param_ho(drive_f, drive_a, drive_p, m=1, w0 = w/2) for w in drive_f]
    evolS5_ref = [p.evol_to_t(t=time, x0 = 0.05, dx0 = 0, energy=True) for p in phoS5_ref]
    
    en_main5 = evolS5_ref[0][2]
    en_total5 = np.sum([e[2] for e in evolS5_ref], 0)
    plt.plot(time, en_main5)
    plt.plot(time, en_total5)    
    cfS5_ref = np.max(en_total5)
    print(cfS5_ref)
    0.71043255
    bounds = [(-0.5, 0.5), (-0.5, 0.5),(-0.5, 0.5), (-0.5, 0.5), (0, 2 * np.pi),(0, 2*np.pi),(0, 2 * np.pi),(0, 2*np.pi)]
    f_multi5 = functools.partial(costfunction_multi_nHh, list_pho=phoS5_ref,time=time)
    de_optim5 = optim.differential_evolution(f_multi5, bounds)


    #Optimized 2 extra harmonics
    drive_a = [0.2,  -0.1556261 ,  0.15473844,  0.11013981, -0.10598907]
    drive_p = [0., 2.61750446, 0.71043255,  4.91648615,  4.29076598]
    phoS_optim5 = [param_ho(drive_f, drive_a, drive_p, m=1, w0 = w/2) for w in drive_f]
    evol_optim5 = [p.evol_to_t(t=time, x0=0.05, dx0 = 0, energy = True) for p in phoS_optim5]

    for er, eo, d in zip(evolS5_ref, evol_optim5, drive_f):
        plt.figure()
        plt.plot(time, er[2]/1000000)
        plt.plot(time, eo[2]/1000000)
        plt.title(str(d))
        plt.ylim([0, 0.001])
        
    plt.figure()
    plt.bar(drive_f, drive_a, width=0.1)
    
    drive_optim = phoS_optim5[0].get_drive(time)
    drive_ref = phoS5_ref[0].get_drive(time)
    plt.figure()
    plt.plot(time, drive_ref)
    plt.plot(time, drive_optim)
        
    

import scipy.integrate as integrate
drive = lambda x: phoS_optim5[0].get_drive(time)
H0_f = lambda x: 

result = integrate.quad(lambda x: special.jv(2.5,x), 0, 4.5)








 # ------------- #
    # STUDY 10
    # Optim 5 phos and 5 harmonics
    # ------------- #
    
    #Model initial
    drive_f = [20, 19.95, 20.05, 19.9, 20.1, 20.15,19.85]
    drive_a = [0.2, 0., 0., 0., 0., 0., 0.]
    drive_p = [0., 0., 0., 0., 0., 0., 0.]
    phoS5_ref = [param_ho(drive_f, drive_a, drive_p, m=1, w0 = w/2) for w in drive_f]
    evolS5_ref = [p.evol_to_t(t=time, x0 = 0.05, dx0 = 0, energy=True) for p in phoS5_ref]
    
    en_main5 = evolS5_ref[0][2]
    en_total5 = np.sum([e[2] for e in evolS5_ref], 0)
    plt.plot(time, en_main5)
    plt.plot(time, en_total5)    
    cfS5_ref = np.max(en_total5)
    print(cfS5_ref)
    bounds = [(-0.5, 0.5), (-0.5, 0.5),(-0.5, 0.5), (-0.5, 0.5),(-0.5, 0.5), (-0.5, 0.5), (0, 2 * np.pi),(0, 2*np.pi), (0, 2 * np.pi),(0, 2*np.pi),(0, 2 * np.pi),(0, 2*np.pi)]
    f_multi5 = functools.partial(costfunction_multi_nHh, list_pho=phoS5_ref,time=time)
    de_optim5 = optim.differential_evolution(f_multi5, bounds)


    #Optimized 2 extra harmonics
    drive_a = [0.2,  -0.1556261 ,  0.15473844,  0.11013981, -0.10598907]
    drive_p = [0., 2.61750446, 0.71043255,  4.91648615,  4.29076598]
    phoS_optim5 = [param_ho(drive_f, drive_a, drive_p, m=1, w0 = w/2) for w in drive_f]
    evol_optim5 = [p.evol_to_t(t=time, x0=0.05, dx0 = 0, energy = True) for p in phoS_optim5]

    for er, eo, d in zip(evolS5_ref, evol_optim5, drive_f):
        plt.figure()
        plt.plot(time, er[2]/1000000)
        plt.plot(time, eo[2]/1000000)
        plt.title(str(d))
        plt.ylim([0, 0.001])
        
    plt.figure()
    plt.bar(drive_f, drive_a, width=0.1)
    
    drive_optim = phoS_optim5[0].get_drive(time)
    drive_ref = phoS5_ref[0].get_drive(time)
    plt.figure()
    plt.plot(time, drive_ref)
    plt.plot(time, drive_optim)
        