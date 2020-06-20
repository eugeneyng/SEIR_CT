import glob
import matplotlib.pyplot as plt
import numpy
import os
import pandas
import scipy
import scipy.integrate
import sympy

# GLOBAL CONSTANTS
delta = 2.703e-5 # Natural Birth Rate (unrelated to disease)
mu = 2.403e-5 # Natural Death Rate (unrelated to disease)
sigma = 0.2 # Average Incubation Period [days^-1]
beds = 1739 # Number of hospital beds available
icub = 100 # Number of ICU beds available

def main():

    # INITIAL CONDITIONS
    beta0 = 0.175 # Rate of Exposure
    N0 = 3.57e6 # Total Population of CT [persons]
    gamma0 = 0.07 # Rate of Removal [days^-1]
    cfr0 = 0.022 # Case Fatality Rate [1.4% (NY) - 3%]
    R0 = (beta0*sigma)/((mu+gamma0)*(mu+sigma)) # Basic Reproduction Number

    c0 = 41762 # Total Number of Cases [persons]
    d0 = 3868  # Number of Deaths [persons]
    i0 = 30383/N0 # Active Infected []
    r0 = (c0-d0-i0)/N0 # Recovered []
    e0 = (R0*i0)/N0  # Exposed []
    s0 = (N0-c0-d0-r0)/N0 # Susceptible []

    # ODE VARIABLES
    tf = 365
    dt = 1
    tspan = [0, tf]
    teval = numpy.arange(0, tf, dt)

    # INIT
    init = [N0, c0, d0, s0, e0, i0, r0, beta0, gamma0, cfr0]

    # SOLVE ODE
    solution = scipy.integrate.solve_ivp(rhs, tspan, init, t_eval=teval)
    solution = numpy.c_[teval, numpy.transpose(solution['y'])]
    # TODO: calculate R0 here
    # solution['R0'] =
    numpy.savetxt('solution.csv', solution, header='Time, Population, Cases, Dead, Susceptible, Exposed, Infected, Removed, beta, gamma, CFR')

    fig = plt.figure()
    fig.add_subplot(1,2,1)
    plt.plot(teval, solution[:,4], label='Susceptible')
    plt.plot(teval, solution[:,5], label='Exposed')
    plt.plot(teval, solution[:,6], label='Infected')
    plt.plot(teval, solution[:,7], label='Removed')
    plt.legend()

    fig.add_subplot(1,2,2)
    plt.plot(teval, solution[:,1], label="Population")
    plt.plot(teval, solution[:,2], label="Cases")
    plt.plot(teval, solution[:,3], label="Dead")
    plt.legend()

    plt.show()

def control():
    pass

def estimate():
    data = pandas.read_csv('nytimes.csv')
    ct_data = data[(data.state == "Connecticut")] # Total number of cases, and number of deaths, from 1/21
    print(ct_data)

def dxdt(x, t=0):
    return numpy.array([ 0.07 * x[1],
                        0.175 * numpy.exp( (-0.175/0.07) * x[0] ) - (0.07 * x[1]) ])

def phase():
    r = numpy.linspace(0, 1, 20)
    i = numpy.linspace(0, 1, 20)
    R, I = numpy.meshgrid(r,i)
    dR, dI = dxdt([R,I])
    M = (numpy.hypot(dR, dI))
    M [ M == 0 ] = 1.
    dR /= M
    dI /= M
    Q = plt.quiver(R, I, dR, dI, M, pivot='mid')
    plt.show()

def rhs(dt, init):

    # Unpack Arguments
    N = init[0]
    c = init[1]
    d = init[2]
    s = init[3]
    e = init[4]
    i = init[5]
    r = init[6]
    beta = init[7]
    gamma = init[8]
    cfr = init[9]

    # Solve dynamics
    sdot = delta - (beta*s*i) - (mu*s)
    edot = (beta*s*i) - (mu*e) - (sigma*e)
    idot = (sigma*e) - (gamma*i) - (mu*i)
    rdot = (gamma*i) - (mu*r)
    betadot = 0
    sigmadot = 0
    gammadot = 0
    cfrdot = 0

    # Track variables
    Ndot = (delta*N) - (mu*N) - (cfr*rdot*N)
    ddot = cfr*rdot*N
    cdot = edot*N

    return [Ndot, cdot, ddot, sdot, edot, idot, rdot, betadot, gammadot, cfrdot]

if __name__ == "__main__":
    # main()
    # estimate()
    phase()
