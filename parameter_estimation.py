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
# pandas.set_option('display.max_columns', None)

# GLOBAL CONSTANTS
delta = 2.703e-5 # Natural Birth Rate (unrelated to disease)
gamma = 0.07 # Rate of Removal [days^-1]
mu = 2.403e-5 # Natural Death Rate (unrelated to disease)
sigma = 0.2 # Average Incubation Period [days^-1]

N = 3565287 # Total Population of CT [persons]
cfr = 0.022 # Case Fatality Rate [1.4% (NY) - 3%]
# R = (beta*sigma)/((mu+gamma)*(mu+sigma)) # Basic Reproduction Number

beds = 8798 # Number of hospital beds available
icub = 674 # Number of ICU beds available

def estimate():
    nyt = pandas.read_csv('data/nytimes.csv')
    nytct = nyt[(nyt.state == "Connecticut")].copy()
    nytct['date'] = pandas.to_datetime(nytct['date'])
    nytct['difference'] = nytct['cases'].diff()
    nytct['pct_change'] = nytct['cases'].pct_change()
    nytct = nytct.reset_index()
    nytct.index += 0

    if os.path.isfile('data/jhuct.pkl'):
        # print("Reading saved JHU data")
        jhuct = pandas.read_pickle('data/jhuct.pkl')
    else:
        filelist = os.listdir('data/csse')
        filelist.sort()
        jhuct = pandas.DataFrame()
        for file in filelist:
            jhu = pandas.read_csv('data/csse/' + file)
            jhuct = jhuct.append(jhu[(jhu.Province_State == "Connecticut")])
        jhuct.to_pickle('data/jhuct.pkl')
    jhuct['date'] = pandas.to_datetime(jhuct['Last_Update'])

    # ODE VARIABLES
    tf = 360
    dt = 1
    tspan = [0, tf]
    teval = numpy.arange(0, tf, dt)

    # SEARCH FOR SMALLEST ERROR IN SIMULATIONS
    base_error = 10
    for beta in ranger(0.1, 0.5, 0.01):
        for e in ranger(0, 1000, 50):
            for i in ranger(0, 1000, 50):

                # INITIAL CONDITIONS
                e0 = e/N        # Exposed
                i0 = i/N        # Active Infected
                r0 = 0/N        # Recovered
                s0 = 1-e0-i0-r0 # Susceptible

                # INIT
                init = [s0, e0, i0, r0, beta]

                # SOLVE ODE
                solution = scipy.integrate.solve_ivp(rhs, tspan, init, t_eval=teval)
                solution = numpy.c_[numpy.transpose(solution['y'])]
                seir = pandas.DataFrame(data=solution)
                seir['date'] = pandas.date_range(start='3/8/2020', periods=len(seir), freq='D')
                seir.columns = ['susceptible', 'exposed', 'infected', 'recovered', 'beta', 'date']

                error = (nytct['cases']/N).head(tf).subtract(seir['infected']).abs().sum()
                if (error < base_error):
                    base_error = error
                    betasearch = beta
                    esearch = e
                    isearch = i

    # Now that we have refined, print and show
    print(betasearch)
    print(esearch)
    print(isearch)

    # esearch = 150
    # isearch = 0
    # betasearch = 0.45

    # INITIAL CONDITIONS
    e0 = esearch/N  # Exposed
    i0 = isearch/N  # Active Infected
    r0 = 0          # Recovered
    s0 = 1-e0-i0-r0 # Susceptible

    # INIT
    init = [s0, e0, i0, r0, betasearch]

    # SOLVE ODE
    solution = scipy.integrate.solve_ivp(rhs, tspan, init, t_eval=teval)
    solution = numpy.c_[numpy.transpose(solution['y'])]
    seir = pandas.DataFrame(data=solution)
    seir['date'] = pandas.date_range(start='3/8/2020', periods=len(seir), freq='D')
    seir.columns = ['susceptible', 'exposed', 'infected', 'recovered', 'beta', 'date']

    fig = plt.figure()
    plt.xlabel("Days")
    plt.ylabel("Portion of Population in Compartment")
    plt.grid(True)
    plt.plot(seir['date'], seir['susceptible'], label='Susceptible')
    plt.plot(seir['date'], seir['exposed'], label='Exposed')
    plt.plot(seir['date'], seir['infected'], label='Simulated Infected')
    plt.plot(seir['date'], seir['recovered'], label='Recovered')
    plt.plot(nytct['date'].head(tf), nytct['cases'].head(tf)/N, label='NYT Infected')
    plt.gcf().autofmt_xdate()
    plt.legend()
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

    return [sdot, edot, idot, rdot, 0]

def ranger(start, end, step):
    while start <= end:
        yield start
        start += step

if __name__ == "__main__":
    estimate()
