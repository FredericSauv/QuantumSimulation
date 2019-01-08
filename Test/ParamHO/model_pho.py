"""
Created on Jan 8 19:58:08 2019
@author: FS
"""
from scipy.integrate import ode
import numpy as np
#import functools
import matplotlib.pylab as plt
from matplotlib import rc
rc('text', usetex=True)

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
        self.drive_f = drive_f
        self.drive_a = drive_a
        self.drive_p = drive_p
        self.w0 = w0
        self.m = m

    def get_kin_en(self, X):
        """ Get the kinetic energy of the system at time t
        """        
        return 1/2 * self.m * X[1]**2
        
    def get_pot_en(self, X, t):
        """ Get the potential energy of the system at time t
        """        
        drive = self.get_drive(self, t)
        return 0.5 * drive * (X[0] ** 2)
           
        
    def get_en(self, X, t):
        """ Get the total energy of the system at time t
        """        
        return self.get_kin_en(X) + self.get_pot_en(X, t)
        
    def get_drive(self, t):
        """ Drive at time t
        """
        return np.sum(self.drive_a * np.cos(self.drive_f*t + self.drive_p))
        
        
    def evol_to_t(self, t, x0 = 0, dx0 = 0.1):
        """ return [X(t_0), ... X(t_n)] where X(t) = [x(t), p(t)]
        shape = (2 x len(t))
        """
        times = [t] if np.ndim(t) == 0 else t
        evol = param_ho.evolve_ode(self.derivative, [x0, dx0], t0=0, times=times)
        return evol
    
    def derivative(self, t, X, omega):
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


if __name__ == '__main__':
    w_int = 1
    drive_f = [2]
    drive_a = [0.2]
    drive_p = [0]
    pho = param_ho(drive_f, drive_a, drive_p, m=1, w0 = w_int)
    pho.evol_to_t(t=10.0)







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
