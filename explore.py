import matplotlib.pyplot as plt
import numpy
import os
import pandas
import scipy
import scipy.integrate
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)

N = 3565287 # Total Population of CT [persons]

def explore():

    if os.path.isfile('data/jhuct.pkl'):
        print("Reading saved JHU data")
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

    plt.plot(jhuct['date'], jhuct['Confirmed']/N, 'g-', label='Cumulative Cases')
    plt.plot(jhuct['date'], jhuct['Active']/N, 'r-', label='Infected')
    plt.plot(jhuct['date'], jhuct['Deaths']/N + jhuct['Recovered']/N, 'k-', label='Recovered')

    plt.xlabel("Date")
    plt.ylabel("Portion of Population in Compartment")
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    explore()
