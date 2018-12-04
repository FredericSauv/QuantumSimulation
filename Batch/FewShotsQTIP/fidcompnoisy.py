#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:12:05 2018

@author: agp1
"""
import pdb
import numpy as np
import qutip.control.fidcomp as qtrlfidcomp
from qutip import Qobj
import qutip.logging_utils as logging
logger = logging.get_logger()

class FidCompUnitNoisy(qtrlfidcomp.FidCompUnitary):
    """
    Customised fidelity computer based on the Unitary fidelity computer
    Adds some noise to the fidelity error
    Different type of noise are implemented:
         None:       No noise
        'GAUSS':     Gaussian noise ~ N(noise_mean, noise_std)
        'SHOT2TGT':  noise_n_meas projective measurements are performed on the state target
        'SHOT2BASIS' noise_n_meas projective measurements are performed on a provided
                     basis (noise_b_meas)
                     
                     
    Questions to answer:
        Should we directly implement the differ
        How to deal with the gradients
        What is the diff when dealing with operator trace then abs: does it make sense
        Remove the XXX_curent test
        No caching implemented so far
    """

    def __init__(self, dyn, params=None):
        super(FidCompUnitNoisy, self).__init__(dyn, params)
        #new parameters relating to noise in the FoM
        self.noise_type = 'GAUSS'
        self.noise_mean = 0.0
        self.noise_std = 1e-10
        self.noise_n_meas = 1
        self.noise_b_meas = None

    def get_noisy_fidelity(self):
        """ (potentially) noisy fidelity.
        Remarks that it is based on the square version of the fidelity ~ proba
        """
        self.perfect_fid = self.get_fidelity_perfect()

        if(self.noise_type is None):
            noisy_fid = self.perfect_fid
            logger.info("No noise, fid (gaussian noise): {}".format(noisy_fid))
            
        if(self.noise_type == 'GAUSS'):
            noisy_fid = self.apply_noise_gauss_abs(self.perfect_fid)
            logger.info("fid (gaussian noise): {}, fid (actual): {}".format(noisy_fid,
                                                          self.perfect_fid))
        elif(self.noise_type == 'SHOT2TGT'):
            #pdb.set_trace()
            noisy_fid = self.get_fidelity_shot2tgt()
            logger.info("fid (shot2tgt): {}, fid (actual): {}".format(noisy_fid,
                                                          self.perfect_fid))
        elif(self.noise == 'SHOT2BASIS'):
            noisy_fid = self.get_fidelity_shot2basis()
            logger.info("fid (shot2tgt): {}, fid (actual): {}".format(noisy_fid,
                                                          self.perfect_fid))
            
        else:
            raise NotImplementedError("type of noise {} not recognized".format(self.noise))
            
        self.noisy_fid = np.clip(noisy_fid, 0,1)
        return self.noisy_fid

    def get_fidelity_perfect(self):
        """ Square version of the fidelity"""
        return np.square(self.get_fidelity())

    def get_fidelity_shot2tgt(self):
        """ Gets the current fidelity value based on SHOT MEASUREMENTS ON THE 
        TARGET STATE prior to normalisation.        
        + NOT CACHED SHOULD BE LOOKED AT
        """
        dyn = self.parent
        k = dyn.tslot_computer._get_timeslot_for_fidelity_calc()
        dyn.compute_evolution()
        if dyn.oper_dtype == Qobj:
            proj = self.fid_norm_func((dyn._onto_evo[k]*dyn._fwd_evo[k]).diag())
            proba = np.square(np.abs(proj))
        else:
            proj = self.fid_norm_func(np.diag(dyn._onto_evo[k].dot(dyn._fwd_evo[k])))
            proba = np.abs(np.square(proj))
        f = np.average(np.random.binomial(self.noise_n_meas, proba)/self.noise_n_meas)
        self.fidelity_shot2tgt_prenorm = f
        if dyn.stats is not None:
                dyn.stats.num_fidelity_computes += 1
        return self.fidelity_shot2tgt_prenorm

    def get_fidelity_shot2basis(self):
        """ Gets the current fidelity value based on SHOT MEASUREMENTS ON A 
        SPECIFIED BASIS prior to normalisation.        
        + NOT CACHED SHOULD BE LOOKED AT
        + 
        """
        if self.noise_b_meas is None:
            raise NotImplementedError('noise_b_meas should be provided')
        dyn = self.parent
        k = dyn.tslot_computer._get_timeslot_for_fidelity_calc()
        dyn.compute_evolution()
        
        #not really nice implementation
        proba = np.zeros((len(self.noise_b_meas)))
        for i, b in enumerate(self.noise_b_meas):
            if isinstance(b, Qobj):
                proj = self.fid_norm_func((b.trans() * dyn._fwd_evo[k]).diag())
                proba[b] = np.square(np.abs(proj))
            else:
                proj = self.fid_norm_func(np.dot(np.conjugate(b), dyn._fwd_evo[k]))
                proba[b] = np.square(np.abs(proj))
        f = np.average(np.random.binomial(self.noise_n_meas, proba)/self.noise_n_meas)
        self.fidelity_shot2basis_prenorm = f
        if dyn.stats is not None:
                dyn.stats.num_fidelity_computes += 1
        return self.fidelity_shot2basis_prenorm

    def apply_noise_gauss_abs(self, ferr_act):
        """ Apply Gaussian noise (corrected for the cases it fells outside [0,1]) 
        """
        err = np.random.normal(self.noise_mean, self.noise_std)
        ferry = ferr_act + err
        if ferry > 1:
            ferry = 2 - ferry
        return np.abs(ferry)

    def get_fid_err_gradient(self):
        """
        Returns the normalised gradient of the fidelity error
        in a (nTimeslots x n_ctrls) array
        The gradients are cached in case they are requested
        mutliple times between control updates
        (although this is not typically found to happen)
        """
        raise NotImplementedError("fidelity gradient related functions haven't been implemented (yet)")
#        if not self.fid_err_grad_current:
#            dyn = self.parent
#            grad_prenorm = self.compute_fid_grad()
#            if self.log_level <= logging.DEBUG_INTENSE:
#                logger.log(logging.DEBUG_INTENSE, "pre-normalised fidelity "
#                           "gradients:\n{}".format(grad_prenorm))
#            # AJGP: Note this check should not be necessary if dynamics are
#            #       unitary. However, if they are not then this gradient
#            #       can still be used, however the interpretation is dubious
#            if self.get_fidelity() >= 1:
#                self.fid_err_grad = self.grad_norm_func(grad_prenorm)
#            else:
#                self.fid_err_grad = -self.grad_norm_func(grad_prenorm)
#
#            self.fid_err_grad_current = True
#            if dyn.stats is not None:
#                dyn.stats.num_grad_computes += 1
#
#            self.grad_norm = np.sqrt(np.sum(self.fid_err_grad**2))
#            if self.log_level <= logging.DEBUG_INTENSE:
#                logger.log(logging.DEBUG_INTENSE, "Normalised fidelity error "
#                           "gradients:\n{}".format(self.fid_err_grad))
#
#            if self.log_level <= logging.DEBUG:
#                logger.debug("Gradient (sum sq norm): "
#                             "{} ".format(self.grad_norm))
#
#        return self.fid_err_grad

    def compute_fid_grad(self):
        """
        Calculates exact gradient of function wrt to each timeslot
        control amplitudes. Note these gradients are not normalised
        These are returned as a (nTimeslots x n_ctrls) array
        """
        raise NotImplementedError("fidelity gradient related functions haven't been implemented (yet)")
#        dyn = self.parent
#        n_ctrls = dyn.num_ctrls
#        n_ts = dyn.num_tslots
#
#        # create n_ts x n_ctrls zero array for grad start point
#        grad = np.zeros([n_ts, n_ctrls], dtype=complex)
#
#        dyn.tslot_computer.flag_all_calc_now()
#        dyn.compute_evolution()
#
#        time_st = timeit.default_timer()
#        # loop through all ctrl timeslots calculating gradients
#        for j in range(n_ctrls):
#            for k in range(n_ts):
#                fwd_evo = dyn._fwd_evo[k]   
#                onto_evo = dyn._onto_evo[k+1]
#                if dyn.oper_dtype == Qobj:
#                    g = (onto_evo*dyn._get_prop_grad(k, j)*fwd_evo).tr()
#                else:
#                    g = qtrlfidcomp._trace(onto_evo.dot(
#                                dyn._get_prop_grad(k, j)).dot(fwd_evo))
#                grad[k, j] = g
#        if dyn.stats is not None:
#            dyn.stats.wall_time_gradient_compute += \
#                timeit.default_timer() - time_st
#        return grad
