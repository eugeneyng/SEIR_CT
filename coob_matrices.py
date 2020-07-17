import control
import numpy

# GLOBAL CONSTANTS
beta = 0.175
delta = 2.703e-5 # Natural Birth Rate (unrelated to disease)
gamma = 0.07 # Rate of Removal [days^-1]
mu = 2.403e-5 # Natural Death Rate (unrelated to disease)
sigma = 0.2 # Average Incubation Period [days^-1]

N = 3.57e6 # Total Population of CT [persons]
cfr = 0.022 # Case Fatality Rate [1.4% (NY) - 3%]
R = (0.175*sigma)/((mu+gamma)*(mu+sigma)) # Basic Reproduction Number

def controlobserve():
    J_dfe = numpy.matrix([
            [0, 0,      -beta,  0],
            [0, -sigma, beta,   0],
            [0, sigma,  -gamma, 0],
            [0, 0,      gamma,  0]
    ])

    K_dfe = numpy.matrix([
            [0],
            [0],
            [0],
            [0]
    ])

    L_dfe = numpy.matrix([
            [1, 1, 1, 1]
    ])

    C_dfe = control.ctrb(J_dfe, K_dfe)
    O_dfe = control.obsv(J_dfe, L_dfe)

    rankC_dfe = numpy.linalg.matrix_rank(C_dfe)
    print(C_dfe)
    rankO_dfe = numpy.linalg.matrix_rank(O_dfe)
    print(O_dfe)

    J_end = numpy.matrix([
            [0, 0,      -gamma, 0],
            [0, -sigma, gamma,  0],
            [0, sigma,  -gamma, 0],
            [0, 0,      gamma,  0]
    ])

    K_end = numpy.matrix([
            [0],
            [0],
            [0],
            [0]
    ])

    L_end = numpy.matrix([
            [1, 1, 1, 1]
    ])

    C_end = control.ctrb(J_end, K_end)
    O_end = control.obsv(J_end, L_end)

    rankC_end = numpy.linalg.matrix_rank(C_end)
    print(C_end)
    rankO_end = numpy.linalg.matrix_rank(O_end)
    print(O_end)


if __name__ == "__main__":
    controlobserve()
