import glob
import matplotlib.pyplot as plt
import numpy
import os
import pandas
import scipy
import scipy.integrate
import sympy

beta0 = 0.175
gamma0 = 0.07

def dRIdt(x, t=0):
    return numpy.array([ gamma0 * x[1],
                        beta0 * numpy.exp( (-beta0/gamma0) * x[0] ) - (gamma0 * x[1]) ])

def phaseRI():
    # see https://scipy-cookbook.readthedocs.io/items/LoktaVolterraTutorial.html
    r = numpy.linspace(0, 1, 50)
    i = numpy.linspace(0, 1, 50)
    R, I = numpy.meshgrid(r,i)
    dR, dI = dRIdt([R,I])
    M = (numpy.hypot(dR, dI))
    M [ M == 0 ] = 1.
    dR /= M
    dI /= M
    # plt.quiver(R, I, dR, dI, M, pivot='mid')
    plt.streamplot(R, I, dR, dI)
    plt.xlabel("Recovered")
    plt.ylabel("Infected")
    plt.show()

if __name__ == "__main__":
    phaseRI()
