# Parameter Estimation

import matplotlib.pyplot as plt
import numpy
import os
import pandas
import scipy
import scipy.integrate

# GLOBAL CONSTANTS
delta   = 2.703e-5 # Natural Birth Rate (unrelated to disease)
gamma   = 0.07 # Rate of Removal [days^-1]
mu      = 2.403e-5 # Natural Death Rate (unrelated to disease)
sigma   = 0.2 # Average Incubation Period [days^-1]
N       = 3565287 # Total Population of CT [persons]
cfr     = 0.022 # Case Fatality Rate [1.4% (NY) - 3%]
beds    = 8798 # Number of hospital beds available
icub    = 674 # Number of ICU beds available

def estimate():

    nyt     = pandas.read_csv('data/nytimes.csv')
    nytct   = nyt[(nyt.state == "Connecticut")].copy()
    nytct['date']       = pandas.to_datetime(nytct['date'])
    nytct['difference'] = nytct['cases'].diff()
    nytct['pct_change'] = nytct['cases'].pct_change()
    nytct   = nytct.reset_index()
    nytct.index += 0

    # ODE VARIABLES
    tf      = 360
    dt      = 1
    tspan   = [0, tf]
    teval   = numpy.arange(0, tf, dt)
    targs   = [tf, dt, tspan, teval]

    (betasearch, esearch, isearch) = search(targs, nytct)
    # (betasearch, esearch, isearch) = (0.42, 100, 50) # Results for tf = 30
    # (betasearch, esearch, isearch) = (0.16, 1650, 950) # Results for tf = 60
    # (betasearch, esearch, isearch) = (0.11, 950, 1950) # Results for tf = 150
    R = (betasearch*sigma)/((mu+gamma)*(mu+sigma)) # Basic Reproduction Number
    print(betasearch, esearch, isearch, R)

    # INITIAL CONDITIONS
    e0 = esearch/N  # Exposed
    i0 = isearch/N  # Active Infected
    r0 = 0          # Recovered (effectively 0)
    s0 = 1-e0-i0-r0 # Susceptible

    # INIT
    init = [s0, e0, i0, r0, betasearch]
    beta = numpy.ones(tf) * betasearch

    # SOLVE ODE
    solution = scipy.integrate.solve_ivp(rhs, tspan, init, t_eval=teval)
    solution = numpy.c_[numpy.transpose(solution['y'])]
    seir            = pandas.DataFrame(data=solution)
    seir['date']    = pandas.date_range(start='3/8/2020', periods=len(seir), freq='D')
    seir.columns    = ['susceptible', 'exposed', 'infected', 'recovered', 'beta', 'date']

    plt.xlabel("Days Since Start of Simulation")
    plt.ylabel("Portion of Population in Compartment")
    plt.xticks(numpy.arange(0, tf, 30))
    plt.yticks(numpy.arange(0, 1, 0.05))
    plt.grid(True)
    plt.plot(seir.index, seir['susceptible'], 'g-', label='Susceptible')
    plt.plot(seir.index, seir['exposed'], 'b-', label='Exposed')
    plt.plot(seir.index, seir['infected'], 'r-', label=r'Infected, $\beta$=0.16')
    plt.plot(nytct.index, nytct['cases']/N, label='Actual Infected, per NYT/JHU')
    plt.plot(seir.index, seir['recovered'], 'k-', label='Recovered')
    plt.plot(seir.index, beta, 'm-', label=r'$\beta$')

    # plt.plot(seir['date'], seir['infected'], label=r'Proportion of Infected, Simulated with $\beta$=0.11')
    # plt.scatter(nytct['date'].head(tf), nytct['cases'].head(tf)/N, label='Proportion of Infected, per NYT/JHU')
    # plt.gcf().autofmt_xdate()
    plt.legend()
    plt.tight_layout()
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

def search(targs, comparison_data):
    tf = targs[0]
    dt = targs[1]
    tspan = targs[2]
    teval = targs[3]

    # SEARCH FOR SMALLEST ERROR IN SIMULATIONS
    base_error = 10
    for beta in numpy.arange(0.1, 0.6, 0.01):
        for e in numpy.arange(0, 2000, 50):
            for i in numpy.arange(0, 2000, 50):
                print(beta, e, i)
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

                error = (((comparison_data['cases']/N).head(tf).subtract(seir['infected']))**2).sum()
                if (error < base_error):
                    print("New Minimum Error")
                    base_error = error
                    betasearch = beta
                    esearch = e
                    isearch = i

    return(betasearch, esearch, isearch)

if __name__ == "__main__":
    estimate()
