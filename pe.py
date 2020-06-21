# Parameter Estimation

import glob
import matplotlib.pyplot as plt
import numpy
import os
import pandas
import scipy
import scipy.integrate
import sympy

def estimate():
    nyt = pandas.read_csv('data/nytimes.csv')
    nytct = nyt[(nyt.state == "Connecticut")]
    nytct['date'] = pandas.to_datetime(nytct['date'])

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

    ax1 = plt.gca()
    jhuct.plot(x='date', y='Confirmed', kind='line', ax=ax1, label='JHU Confirmed')
    jhuct.plot(x='date', y='Recovered', kind='line', ax=ax1, label='JHU Recovered')
    nytct.plot(x='date', y='cases', kind='line', ax=ax1, label='NYT Confirmed')
    plt.show()

if __name__ == "__main__":
    estimate()
