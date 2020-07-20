# Parameter Estimation

import glob
import matplotlib.pyplot as plt
import numpy
import os
import pandas
import scipy
import scipy.integrate
import sympy
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)

# GLOBAL CONSTANTS
beta = 0.175 # Rate of Exposure
delta = 2.703e-5 # Natural Birth Rate (unrelated to disease)
gamma = 0.07 # Rate of Removal [days^-1]
mu = 2.403e-5 # Natural Death Rate (unrelated to disease)
sigma = 0.2 # Average Incubation Period [days^-1]

N = 3565287 # Total Population of CT [persons]
cfr = 0.022 # Case Fatality Rate [1.4% (NY) - 3%]
R = (beta*sigma)/((mu+gamma)*(mu+sigma)) # Basic Reproduction Number

beds = 1739 # Number of hospital beds available
icub = 100 # Number of ICU beds available

def estimate():
    nyt = pandas.read_csv('data/nytimes.csv')
    nytct = nyt[(nyt.state == "Connecticut")]
    nytct['date'] = pandas.to_datetime(nytct['date'])
    nytct = nytct.reset_index()
    nytct.index += 0

    # if os.path.isfile('data/jhuct.pkl'):
    #     # print("Reading saved JHU data")
    #     jhuct = pandas.read_pickle('data/jhuct.pkl')
    # else:
    #     filelist = os.listdir('data/csse')
    #     filelist.sort()
    #     jhuct = pandas.DataFrame()
    #     for file in filelist:
    #         jhu = pandas.read_csv('data/csse/' + file)
    #         jhuct = jhuct.append(jhu[(jhu.Province_State == "Connecticut")])
    #     jhuct.to_pickle('data/jhuct.pkl')
    # jhuct['date'] = pandas.to_datetime(jhuct['Last_Update'])
    # # print(jhuct)
    #
    # ax1 = plt.gca()
    # jhuct.plot(x='date', y='Active', kind='line', ax=ax1, label='JHU Active')
    # jhuct.plot(x='date', y='Confirmed', kind='line', ax=ax1, label='JHU Confirmed')
    # jhuct.plot(x='date', y='Deaths', kind='line', ax=ax1, label='JHU Deaths')
    # jhuct.plot(x='date', y='People_Hospitalized', kind='line', ax=ax1, label='JHU Hospitalized')
    # jhuct.plot(x='date', y='Recovered', kind='line', ax=ax1, label='JHU Recovered')
    # plt.show()

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
    plt.xlabel("Days")
    plt.ylabel("Portion of Population in Compartment")
    plt.grid(True)
    plt.plot(teval, solution[:,1], label='Susceptible')
    plt.plot(teval, solution[:,2], label='Exposed')
    plt.plot(teval, solution[:,3], label='Infected')
    plt.plot(teval, solution[:,4], label='Removed')
    plt.plot(nytct.index, nytct['cases']/N, label='NYT Infected')
    plt.legend()
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
    estimate()
