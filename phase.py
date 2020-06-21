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

def dxdt(x, t=0):
    return numpy.array([ gamma0 * x[1],
                        beta0 * numpy.exp( (-beta0/gamma0) * x[0] ) - (gamma0 * x[1]) ])

def phase():
    # see https://scipy-cookbook.readthedocs.io/items/LoktaVolterraTutorial.html
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

if __name__ == "__main__":
    phase()
