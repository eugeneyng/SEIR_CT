import glob
import matplotlib.pyplot as plt
import numpy
import os
import pandas
import scipy
import scipy.integrate
import sympy
from gekko import GEKKO

# GLOBAL CONSTANTS
delta = 2.703e-5 # Natural Birth Rate (unrelated to disease)
gamma = 0.07 # Rate of Removal [days^-1]
mu = 2.403e-5 # Natural Death Rate (unrelated to disease)
sigma = 0.2 # Average Incubation Period [days^-1]

N = 3.57e6 # Total Population of CT [persons]
cfr = 0.022 # Case Fatality Rate [1.4% (NY) - 3%]
R = (0.175*sigma)/((mu+gamma)*(mu+sigma)) # Basic Reproduction Number

beds = 1739 # Number of hospital beds available
icub = 100 # Number of ICU beds available

def main():
    tf = 365
    dt = 1

    #   __ _    ___  | | __ | | __   ___
    #  / _` |  / _ \ | |/ / | |/ /  / _ \
    # | (_| | |  __/ |   <  |   <  | (_) |
    #  \__, |  \___| |_|\_\ |_|\_\  \___/
    # |___/

    # Create GEKKO Model
    m = GEKKO(remote=False)
    m.time = numpy.arange(0, tf, dt)

    # Define variables and initial value for GEKKO Model
    m.beta = m.MV(value=0.175, lb=0, ub=1)
    m.s = m.SV(value=(N-250-360-400)/N, lb=0, ub=1)
    m.e = m.CV(value=250/N, lb=0, ub=1)
    m.i = m.CV(value=360/N, lb=0, ub=1)
    m.r = m.SV(value=400/N, lb=0, ub=1)

    m.Equation( m.s.dt() == delta - (m.beta*m.s*m.i) - (mu*m.s) )
    m.Equation( m.e.dt() == (m.beta*m.s*m.i) - (sigma*m.e) - (mu*m.e) )
    m.Equation( m.i.dt() == (sigma*m.e) - (gamma*m.i) - (mu*m.i) )
    m.Equation( m.r.dt() == (gamma*m.i) - (mu*m.r) )

    # TUNING PARAMETERS:
    # https://gekko.readthedocs.io/en/latest/tuning_params.html
    # https://gekko.readthedocs.io/en/latest/global.html

    # MV Tuning
    m.beta.STATUS = 1
    m.beta.FSTATUS = 0
    # m.Tc.DMAX = 0.1
    # m.Tc.DMAXHI = 0.1   # constrain movement up
    # m.Tc.DMAXLO = 0.1 # quick action down

    # CV Tuning
    m.s.FSTATUS = 1
    m.e.STATUS = 1
    m.e.FSTATUS = 0
    m.e.SPHI = 0.05
    m.e.SPLO = 0
    m.i.STATUS = 1
    m.i.FSTATUS = 1
    m.i.SPHI = 0.05
    m.i.SPLO = 0
    m.r.FSTATUS = 1

    # MODEL OPTIONS
    m.options.CV_TYPE = 1
    m.options.IMODE = 6
    m.options.SOLVER = 3

    #   ___     __| |   ___
    #  / _ \   / _` |  / _ \
    # | (_) | | (_| | |  __/
    #  \___/   \__,_|  \___|

    # Define initial values for ODE solver
    s0 = (N-360-250-400)/N  # Susceptible []
    e0 = 250/N              # Exposed []
    i0 = 360/N              # Active Infected []
    isp = 0.1               # Active Infected Setpoint (under 10% of population)
    r0 = 400/N              # Recovered []
    beta0 = 0.175

    t = numpy.arange(0, tf, dt)

    # Store results
    s = numpy.ones(len(t)) * s0
    e = numpy.ones(len(t)) * e0
    i = numpy.ones(len(t)) * i0
    isp = numpy.ones(len(t)) * isp
    r = numpy.ones(len(t)) * r0
    beta = numpy.ones(len(t)) * beta0

    #  ___  (_)  _ __ ___    _   _  | |   __ _  | |_  (_)   ___    _ __
    # / __| | | | '_ ` _ \  | | | | | |  / _` | | __| | |  / _ \  | '_ \
    # \__ \ | | | | | | | | | |_| | | | | (_| | | |_  | | | (_) | | | | |
    # |___/ |_| |_| |_| |_|  \__,_| |_|  \__,_|  \__| |_|  \___/  |_| |_|

    plt.ion()
    plt.show()

    for ind in range(len(t)-1):
        print(ind)
        tsim = [ t[ind], t[ind+1] ]
        init = [ s[ind], e[ind], i[ind], r[ind], beta[ind] ]
        sol = scipy.integrate.solve_ivp(rhs, tsim, init)

        s[ind+1] = sol['y'][0][-1]
        e[ind+1] = sol['y'][1][-1]
        i[ind+1] = sol['y'][2][-1]
        r[ind+1] = sol['y'][3][-1]

        if ind%60 == 0:
            m.e.MEAS = e[ind+1]
            m.i.MEAS = i[ind+1]
            m.solve(disp=True)
            beta[ind+1] = m.beta.NEWVAL
        else:
            beta[ind+1] = beta[ind]

        print(beta[ind+1])

        plt.cla()
        plt.plot(t, s, 'g-')
        plt.plot(t, e, 'b-')
        plt.plot(t, i, 'r-')
        plt.plot(t, r, 'k-')
        plt.pause(0.0001)

    plt.ioff()
    plt.show()

def rhs(dt, init):

    # Unpack Arguments
    s = init[0]
    e = init[1]
    i = init[2]
    r = init[3]
    beta = init[4]

    # Solve dynamics
    sdot = delta - (beta*s*i) - (mu*s)
    edot = (beta*s*i) - (mu*e) - (sigma*e)
    idot = (sigma*e) - (gamma*i) - (mu*i)
    rdot = (gamma*i) - (mu*r)
    betadot = 0

    return [sdot, edot, idot, rdot, betadot]

if __name__ == "__main__":
    main()
