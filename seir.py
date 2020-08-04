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

N = 3565287 # Total Population of CT [persons]
cfr = 0.022 # Case Fatality Rate [1.4% (NY) - 3%]

beds = 8798 # Number of hospital beds available
icub = 674 # Number of ICU beds available

def main():
    tf = 360
    dt = 1
    control_period = 30

    #   __ _    ___  | | __ | | __   ___
    #  / _` |  / _ \ | |/ / | |/ /  / _ \
    # | (_| | |  __/ |   <  |   <  | (_) |
    #  \__, |  \___| |_|\_\ |_|\_\  \___/
    # |___/

    # Create GEKKO Model
    m = GEKKO(remote=False)
    m.time = numpy.arange(0, tf, dt)

    # Define variables and initial value for GEKKO Model
    # m.beta = m.MV(value=0.15, lb=0, ub=1)
    m.betam = m.MV(value=1, lb=0, ub=10, integer=True)
    m.s = m.SV(value=0.969865, lb=0, ub=1)
    m.e = m.CV(value=.006550, lb=0, ub=1)
    m.i = m.CV(value=.010616, lb=0, ub=1)
    m.r = m.SV(value=.013146, lb=0, ub=1)

    # m.Equation( m.s.dt() == delta - (m.beta*m.s*m.i) - (mu*m.s) )
    # m.Equation( m.e.dt() == (m.beta*m.s*m.i) - (sigma*m.e) - (mu*m.e) )
    m.Equation( m.s.dt() == delta - (( (m.betam + 0.5) / 10 )*m.s*m.i) - (mu*m.s) )
    m.Equation( m.e.dt() == (( (m.betam + 0.5) / 10 )*m.s*m.i) - (sigma*m.e) - (mu*m.e) )
    m.Equation( m.i.dt() == (sigma*m.e) - (gamma*m.i) - (mu*m.i) )
    m.Equation( m.r.dt() == (gamma*m.i) - (mu*m.r) )

    # TUNING PARAMETERS:
    # https://gekko.readthedocs.io/en/latest/tuning_params.html
    # https://gekko.readthedocs.io/en/latest/global.html

    # Manipulated Variable
    # m.beta.STATUS = 1
    # m.beta.FSTATUS = 0
    m.betam.STATUS = 1
    m.betam.FSTATUS = 0

    # Controlled Variables
    m.s.FSTATUS = 1
    m.e.FSTATUS = 1
    m.r.FSTATUS = 1

    m.i.STATUS = 1
    m.i.FSTATUS = 1
    m.i.SPHI = 0.2
    m.i.SPLO = 0.05

    # m.Obj(m.i)

    # MODEL OPTIONS

    m.options.IMODE = 6 # CONTROL
    m.options.SOLVER = 1 # APOPT (Advanced Process Optimizer) (Only solver that handles Mixed Integer problems)
    # m.options.SOLVER = 3 # IPOPT (Interior Point Optimizer)

    m.solver_options = ['minlp_gap_tol 10',\
                    'minlp_maximum_iterations 10',\
                    'minlp_max_iter_with_int_sol 5',\
                    'minlp_as_nlp 0',\
                    'minlp_branch_method 1',\
                    'minlp_integer_tol 0.35']

    #   ___     __| |   ___
    #  / _ \   / _` |  / _ \
    # | (_) | | (_| | |  __/
    #  \___/   \__,_|  \___|

    # Define initial values for ODE solver (from parameter_estimation.py)
    s0 = 0.969865   # Susceptible []
    e0 = 0.006550   # Exposed []
    i0 = 0.010616   # Active Infected []
    isp = 0.1      # Active Infected Setpoint (under 10% of population)
    r0 = 0.013146   # Recovered []
    betam0 = 1
    beta0 = (betam0 + 0.5)/10

    t = numpy.arange(0, tf, dt)

    # Store results
    s = numpy.ones(len(t)) * s0
    e = numpy.ones(len(t)) * e0
    i = numpy.ones(len(t)) * i0
    isp = numpy.ones(len(t)) * isp
    r = numpy.ones(len(t)) * r0
    betam = numpy.ones(len(t)) * betam0
    beta = numpy.ones(len(t)) * beta0

    #  ___  (_)  _ __ ___    _   _  | |   __ _  | |_  (_)   ___    _ __
    # / __| | | | '_ ` _ \  | | | | | |  / _` | | __| | |  / _ \  | '_ \
    # \__ \ | | | | | | | | | |_| | | | | (_| | | |_  | | | (_) | | | | |
    # |___/ |_| |_| |_| |_|  \__,_| |_|  \__,_|  \__| |_|  \___/  |_| |_|

    plt.ion()
    plt.show()

    for ind in range(len(t)-1):
        print(ind)
        print(betam[ind])
        print(beta[ind])
        tsim = [ t[ind], t[ind+1] ]
        init = [ s[ind], e[ind], i[ind], r[ind], betam[ind], beta[ind] ]
        sol = scipy.integrate.solve_ivp(rhs, tsim, init)

        s[ind+1] = sol['y'][0][-1]
        e[ind+1] = sol['y'][1][-1]
        i[ind+1] = sol['y'][2][-1]
        r[ind+1] = sol['y'][3][-1]

        if ind%control_period == 0:
            m.e.MEAS = e[ind]
            m.i.MEAS = i[ind]
            m.solve(disp=True)
            # beta[ind+1] = m.beta.NEWVAL
            betam[ind+1] = m.betam.NEWVAL
            beta[ind+1] = (betam[ind+1] + 0.5) / 10
        else:
            # beta[ind+1] = beta[ind]
            betam[ind+1] = betam[ind]
            beta[ind+1] = beta[ind]

        plt.cla()
        plt.xticks(numpy.arange(0, tf, 30))
        plt.yticks(numpy.arange(0, 1, 0.05))
        plt.grid(True)
        plt.xlabel("Days Since Start of Simulation")
        plt.plot(t, s, 'g-', label='Susceptible')
        plt.plot(t, e, 'b-', label='Exposed')
        plt.plot(t, i, 'r-', label='Infectious')
        plt.plot(t, r, 'k-', label='Removed')
        plt.plot(t, beta, 'm-', label=r'$\beta$')
        plt.legend()
        plt.pause(0.0001)

    plt.ioff()
    plt.show()

def rhs(dt, init):

    # Unpack Arguments
    s = init[0]
    e = init[1]
    i = init[2]
    r = init[3]
    betam = init[4]
    beta = init[5]

    # Solve dynamics
    sdot = delta - (beta*s*i) - (mu*s)
    edot = (beta*s*i) - (mu*e) - (sigma*e)
    idot = (sigma*e) - (gamma*i) - (mu*i)
    rdot = (gamma*i) - (mu*r)
    betamdot = 0
    betadot = 0

    return [sdot, edot, idot, rdot, betamdot, betadot]

if __name__ == "__main__":
    main()
