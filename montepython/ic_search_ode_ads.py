# -*- coding: utf-8 -*-
"""
Created on Tue Oct 1 2019

@author: 42173
"""
from __future__ import print_function
from math import exp, expm1, log1p, tanh, atanh, sqrt, cosh, sinh
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar, minimize_scalar, root
from random import random


# the MCMC parameters are scf_f and ln(1+z_c)
# this code shoots for exact scalar field initial condition (phi_i,V_0) which
# reproduces scf_f at z_c

###################
# add support for supergravity potential
# V=V0*(exp(gamma*tanh(phi/root6alpha))-1)^2-(V0-V_Lambda)
###################

###################
# add support for late inflation model
# V=V0*(1-cos(x))^2n+V_Lambda for x<0 and V=V_Lambda for x>0
# in this model z_c stands for onset of the late inflation in radiation dominant era
# scf_f stands for initial x

class scf_ic_finder:

    # input is the desired precision
    def __init__(
            self,
            model='phi4ads',
            n=2,
            alpha=0.1,
            precision=0.01,
            initial_time=-4.6,
            verbose=0):
        # precision and initial time
        self.precision = precision
        self.t_i = initial_time
        # whether print results
        self.verbose = verbose
        # default cosmo parameters
        self.omega_m = 0.142
        self.T_T0 = 1
        # default MCMC parameters
        self.scf_f = 0.07
        self.z_c = exp(8.2)
        self.scf_params = np.zeros(2)
        self.ads = 3.79e-4
        self.alpha = alpha
        # default scf parameters
        self.scf_initial = np.zeros(2)
        self.n = n
        self.model = model
        self.scf_f_bound = 0.3

    # input are standard LambdaCDM parameters: omega_b, omega_cdm, T_cmb, H0
    def set_cosmo_params(
            self,
            omega_m=0.142,
            T0=2.7255,
            H0=70):
        self.omega_m = omega_m
        h = 0.01 * H0
        self.T_T0 = T0 / 2.7255

    # input are MCMC parameters scf_f, ln(1+z_c), ads(in powerlaw), gamma(in alpha-attractor)
    # alpha_ads is defined by V_ads = 3 * Hc^2 * ads
    def set_MCMC_params(
            self,
            scf_f=0.07,
            ln1plusz_c=8.5,
            ads=3.79e-4):
        self.scf_f = scf_f
        self.z_c = expm1(ln1plusz_c)
        self.ads = ads
        if self.verbose > 0:
            print("->Want scf_f = %.3f at z = %.1f" % (self.scf_f, self.z_c))

    def search(self):
        Hc2 = (self.omega_m * (self.z_c + 1) ** 3 +
               0.000042 * (self.T_T0 * (self.z_c + 1)) ** 4) / 2998. / 2998.
        # get the initial guess for V_0 and phi_i
        self.scf_params[0] = 3 ** (self.n + 1) * Hc2 / (
                2 * self.n * (2 * self.n - 1)) ** self.n / (
                                        self.scf_f + self.ads) ** (self.n - 1)
        self.scf_initial[0] = - sqrt(
            2 * self.n * (2 * self.n - 1) / 3. * (self.scf_f + self.ads))
        self.scf_params[1] = 3. * Hc2 * self.ads
        if self.verbose > 0:
            print("->Initial guess V0 = " + str(
                self.scf_params[0]) + ", phi_i = " + str(
                self.scf_initial[0]) + ", Hc2 = " + str(Hc2))

        x0 = np.zeros(2)
        # V0
        x0[0] = self.scf_params[0]
        # phi_i
        x0[1] = self.scf_initial[0]
        ic_search_rlt = root(self.is_ic, x0, tol=self.precision)
        self.scf_params[0] = ic_search_rlt.x[0]
        self.scf_initial[0] = ic_search_rlt.x[1]
        if self.model == 'supergravity':
            CLASS_params = np.array(
                [self.scf_params[0], self.scf_params[1], self.scf_params[2], 0,
                 self.scf_initial[0], self.scf_initial[1]])
        else:
            CLASS_params = np.array(
                [self.scf_params[0], self.scf_params[1], self.alpha, self.alpha,
                 self.scf_initial[0], self.scf_initial[1]])

        #print(CLASS_params)
        if self.verbose > 0:
            print(', '.join(map(str, CLASS_params)))
        return ', '.join(map(str, CLASS_params)), log1p(self.z_c)

    def V(self, phi):
        if self.model == 'supergravity':
            # V=V0*(exp(gamma*tanh(phi/root6alpha))-1)^2-(V0-V_Lambda)
            V0 = self.scf_params[0]
            gamma = self.scf_params[1]
            root6alpha = self.scf_params[2]
            Vlambda = self.scf_params[3]
            return V0 * (
                expm1(gamma * tanh(-phi / root6alpha))) ** 2 - V0 + Vlambda
        else:
            return self.scf_params[0] * \
                   phi ** (2 * self.n) - self.scf_params[1]

    def dV(self, phi):
        if self.model == 'supergravity':
            V0 = self.scf_params[0]
            gamma = self.scf_params[1]
            root6alpha = self.scf_params[2]
            return -2 * exp(- gamma * tanh(phi / root6alpha)) * expm1(
                -(gamma * tanh(phi / root6alpha))) * gamma * V0 / cosh(
                phi / root6alpha) / cosh(phi / root6alpha) / root6alpha
        else:
            return 2 * self.n * self.scf_params[0] * phi ** (2 * self.n - 1)

    def ddV(self, phi):
        if self.model == 'supergravity':
            V0 = self.scf_params[0]
            gamma = self.scf_params[1]
            root6alpha = self.scf_params[2]
            return (-2 * gamma * V0 * pow(cosh(phi / root6alpha), -4) * (
                    (-1 + expm1(gamma * tanh(phi / root6alpha))) * gamma + expm1(
                gamma * tanh(phi / root6alpha)) * sinh((2 * phi) / root6alpha))) / (
                           exp(2 * gamma * tanh(
                               phi / root6alpha)) * root6alpha * root6alpha)
        else:
            return (2 * self.n) * (2 * self.n - 1) * \
                   self.scf_params[0] * phi ** (2 * self.n - 2)

    def H2(self, t, scf_state):
        a_rel = (self.z_c + 1) * exp(-t)
        rho_m = 3 * self.omega_m * a_rel * a_rel * a_rel / 2998. / 2998.
        rho_r = 3 * 0.000042 * (a_rel * self.T_T0) ** 4 / 2998. / 2998.
        return (2 * (rho_r + rho_m + self.V(scf_state[0]))) / (
                6. - scf_state[1] * scf_state[1])

    # the EoM phi'' + (rho-P)/2/H^2*phi' + dV/dphi/H^2 = 0
    # x1 = phi,x2 = phi'
    def equation(self, t, scf_state):
        a_rel = (self.z_c + 1) * exp(-t)
        rho_m = 3 * self.omega_m * a_rel * a_rel * a_rel / 2998. / 2998.
        rho_r = 3 * 0.000042 * (a_rel * self.T_T0) ** 4 / 2998. / 2998.
        x_prime = np.zeros(2)
        x_prime[0] = scf_state[1]
        # rho - P = rho_m + 2/3 rho_r + 2V
        x_prime[1] = - (
                (rho_m + 2. / 3. * rho_r + 2 * self.V(scf_state[0])) / 2. * scf_state[
            1] + self.dV(
            scf_state[0])) / self.H2(
            t, scf_state)
        return x_prime

    def is_ic(self, x):
        rlt = np.zeros(2)
        phi_c = self.scf_initial[0]

        V0_original = self.scf_params[0]
        self.scf_params[0] = x[0]
        x0 = np.array([x[1], 0.])
        # time argument t = ln(a/a_c)
        scf_rlt = solve_ivp(
            self.equation, [
                self.t_i, 0], x0, method='RK23', t_eval=[0]).y.flatten()
        # rolling condition ddV_c=9H_c^2 is equivalent to phi=phi_c, exact ic at rlt[1]=1
        if self.model == 'supergravity':
            rlt[1] = scf_rlt[0] / phi_c
        else:
            rlt[1] = self.ddV(scf_rlt[0]) / self.H2(0, scf_rlt) / 9.
        # scalar field energy fraction rho_c=3fH_c^2, exact ic at rlt[0]=1
        rlt[0] = (scf_rlt[1] * scf_rlt[1] / 2. +
                  self.V(scf_rlt[0]) / self.H2(0, scf_rlt)) / 3. / self.scf_f
        self.scf_params[0] = V0_original

        # exact ic at [0,0]
        return rlt - 1
