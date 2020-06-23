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

def main():

    nyt = pandas.read_csv('data/nytimes.csv')
    nytct = nyt[(nyt.state == "Connecticut")]
    nytct['date'] = pandas.to_datetime(nytct['date'])
    nytct = nytct.reset_index()
    nytct.index += 0

    # INITIAL CONDITIONS
    s0 = (N-360-250-400)/N  # Susceptible []
    e0 = 250/N              # Exposed []
    i0 = 360/N              # Active Infected []
    r0 = 400/N              # Recovered []

    # ODE VARIABLES
    tf = 365
    dt = 1
    tspan = [0, tf]
    teval = numpy.arange(0, tf, dt)

    # INIT
    init = [s0, e0, i0, r0]

    # SOLVE ODE
    solution = scipy.integrate.solve_ivp(rhs, tspan, init, t_eval=teval)
    solution = numpy.c_[teval, numpy.transpose(solution['y'])]
    # TODO: calculate R0 here
    # solution['R0'] =
    numpy.savetxt('solution.csv', solution, header='Time, Susceptible, Exposed, Infected, Removed')

    fig = plt.figure()
    # fig.add_subplot(1,2,1)
    plt.plot(teval, solution[:,1], label='Susceptible')
    plt.plot(teval, solution[:,2], label='Exposed')
    plt.plot(teval, solution[:,3], label='Infected')
    plt.plot(teval, solution[:,4], label='Removed')
    plt.plot(nytct.index, nytct['cases']/N, label='NYT Infected')
    plt.legend()

    # fig.add_subplot(1,2,2)
    # plt.legend()

    plt.show()

def rhs(dt, init):

    # Unpack Arguments
    s = init[0]
    e = init[1]
    i = init[2]
    r = init[3]

    # Solve dynamics
    sdot = delta - (beta*s*i) - (mu*s)
    edot = (beta*s*i) - (mu*e) - (sigma*e)
    idot = (sigma*e) - (gamma*i) - (mu*i)
    rdot = (gamma*i) - (mu*r)

    return [sdot, edot, idot, rdot]

if __name__ == "__main__":
    main()
