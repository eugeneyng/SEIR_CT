# EUGENE NG

import scipy

# Global Constants
N = 3.57e6 # Total Population of CT [persons]
a = 0.2 # Average Incubation Period [days^-1]
beta = 1 # Rate of Exposure
sigma = 1 # Rate at which an exposed person becomes infective [days^-1]
gamma = 0.001 # Rate of Recovery [days^-1] [0.001, 0.05]
cfr = 0.025 # Case Fatality Rate [2-3%]
delta = 9.8 # Natural Birth Rate (unrelated to disease)
mu = 8.7 # Natural Death Rate (unrelated to disease)
R0 = (a/(mu+a))*(beta/(mu+gamma)) # Basic Reproduction Number
t = 365 # How long the model will run
dt = 1 # Step Size

def main():
    t = 41303 # Total Number of Cases [persons]
    d = 3769  # Number of Deaths [persons]
    s = t-d-r # Number of Susceptible [persons]
    e = 0     # Number of Exposed [persons]
    i = 30912 # Number of Active Infected [persons]
    r = t-d-i # Number of Recovered [persons]

def rhs(dt, init):


def dynamics(args):

    # Unpack args
    s = args[0]
    e = args[1]
    i = args[2]
    r = args[3]
