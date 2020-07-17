# This is an midpoint method of solving the ODEs (a refinement of Euler's method)
# Advantage is that it has a local error of O(dt^3) and a global error of O(dt^2) making it similar to RK23
# Another advantage is that it is a fixed timestep solver which can take into account the control at every timestep

import glob
import matplotlib.pyplot as plt
import numpy
import os
import pandas
import scipy
import scipy.integrate
import sympy

# GLOBAL CONSTANTS
beta = 0.175 # Rate of Exposure
delta = 2.703e-5 # Natural Birth Rate (unrelated to disease)
gamma = 0.07 # Rate of Removal [days^-1]
mu = 2.403e-5 # Natural Death Rate (unrelated to disease)
sigma = 0.2 # Average Incubation Period [days^-1]

N = 3.57e6 # Total Population of CT [persons]
cfr = 0.022 # Case Fatality Rate [1.4% (NY) - 3%]
R = (beta*sigma)/((mu+gamma)*(mu+sigma)) # Basic Reproduction Number

beds = 1739 # Number of hospital beds available
icub = 100 # Number of ICU beds available

def midpoint():

    # ODE VARIABLES
    tf = 365
    dt = 1

    s = numpy.zeros(tf)
    e = numpy.zeros(tf)
    i = numpy.zeros(tf)
    r = numpy.zeros(tf)
    t = numpy.arange(0, tf, dt)

    # INITIAL CONDITIONS
    i[0] = 360/N    # Active Infected []
    e[0] = 250/N      # Exposed []
    r[0] = 400/N      # Recovered []
    s[0] = (N-r[0]-e[0]-i[0])/N # Susceptible []

    for index in range(1, tf):
        [fs, fe, fi, fr] = dynamics(s[index-1],
                                    e[index-1],
                                    i[index-1],
                                    r[index-1])
        [ds, de, di, dr] = dynamics(s[index-1] + dt/2*fs,
                                    e[index-1] + dt/2*fe,
                                    i[index-1] + dt/2*fi,
                                    r[index-1] + dt/2*fr)

        s[index] = s[index-1] + dt*ds
        e[index] = e[index-1] + dt*de
        i[index] = i[index-1] + dt*di
        r[index] = r[index-1] + dt*dr

    fig = plt.figure()
    plt.plot(t, s, label='Susceptible')
    plt.plot(t, e, label='Exposed')
    plt.plot(t, i, label='Infected')
    plt.plot(t, r, label='Removed')
    plt.legend()
    plt.show()

def dynamics(s, e, i, r):
    ds = delta - (beta*s*i) - (mu*s)
    de = (beta*s*i) - (mu*e) - (sigma*e)
    di = (sigma*e) - (gamma*i) - (mu*i)
    dr = (gamma*i) - (mu*r)
    return [ds, de, di, dr]

if __name__ == "__main__":
    midpoint()
