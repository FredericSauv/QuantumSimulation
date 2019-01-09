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


if __name__ == '__main__':
    w_int = 10
    t_fin = 20.2
    t_strobo = 2 * np.pi / w_int 
    nt_period = 50
    time = np.arange(0, t_fin, t_strobo / nt_period)
    strobo_ind = np.arange(int(nt_period/6), len(time), nt_period)

    ####### Look at the dynamics with a monochromatic driving
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
            
    
    ####### 3 harmonics 
    drive_f = [20, 19.5, 20.5]
    drive_a = [0.2, 0.25, 0.25]
    drive_p = [0, -0.65, 0.65]
    pho_bi = param_ho(drive_f, drive_a, drive_p, m=1, w0 = w_int)
    evol = pho_bi.evol_to_t(t=time, x0=0.05, dx0 = 0, energy = True)
    toplot = evol[2]
    print(np.max(toplot)/1000000)
    plt.plot(time, toplot)
    plt.scatter(time[strobo_ind], toplot[strobo_ind])



    ####### Optimization 3 harmonics (1 fixed on resonance )
    #Model initial
    drive_f = [20, 19.5, 20.5]
    drive_a = [0.2, 0., 0.]
    drive_p = [0., 0., 0.]
    pho_ref = param_ho(drive_f, drive_a, drive_p, m=1, w0 = w_int)
    evol_ref = pho_ref.evol_to_t(t=time, x0=0.05, dx0 = 0, energy = True)
    cf_ref = np.max(evol_ref[2])
    
    
    #Cost function to minimize
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

    #Optimization
    


#    def optim_drive(self, omega, h_ref = 10.0, noise_h0 = 0.1, nb_repeat = 10, 
#                          s = 40, nb_period = 1, verbose = False, return_to_init = False):
#        """ For a given h_0 and omega how close do we get to href after one cycle
#        i.e. return h(T) - href
#        
#        Parameters
#        ----------
#        h_ref: float 
#            the ideal initial height
#        omega: float
#            driving of the miror
#        noise_h0: float
#            noise in the preparation of the initial position 
#            x0_i ~ N(h_ref, noise_h0^2)
#        nb_repeat: int
#            number of repetition of the experiment
#        nb_period: int
#            number of period (T) after which we measure the height of the
#            atom - with T = s * 2 Pi / omega
#        s: int
#            number of cycles
#        verbose: bool
#            if True print the result
#        return_to_init: <boolean>
#            instead of using h_ref as the reference we use the real h_0 
#            (i.e. h_ref + some noise)
#        
#        Output
#        ------
#            res 
#            = 1/nb_repeat * sum |h(nb_period * T) - h_ref|
#            shape = (len(nb_period), )
#        """
#        big_omega = omega / s
#        T_ref =  2 * np.pi/ big_omega
#        T = nb_period * T_ref if(np.ndim(nb_period) == 0) else T_ref * np.array(nb_period)
#        noise = np.random.normal(0.0, scale = noise_h0, size = nb_repeat)
#        if(return_to_init):
#            res = [np.abs(self.simulate_to_t(omega, T, h_ref*(1+n))[0] - h_ref*(1+n)) for n in noise]
#        else:
#            res = [np.abs(self.simulate_to_t(omega, T, h_ref*(1+n))[0] - h_ref) for n in noise]
#        avg = np.average(res, axis = 0)
#        if(verbose):
#            print(avg)
#        return avg
#    
