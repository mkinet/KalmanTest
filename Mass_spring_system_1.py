"""
This script implements a kalman filter for a simple mass-spring system. Observations are taken as analytic solution +
noise. True and estimated states are plotted.
"""

import numpy as np
import matplotlib.pyplot as plt

from pykalman import KalmanFilter

def analytic_solution(k,m,t):
    """
    Compute analytic solution of mass spring system.

    """
    return np.sin(np.sqrt(k/m)*t)

def add_noise(x,sigma):
    """
    Add gaussian noise of variance sigma to time-serie x.
    :param x:
    :param sigma:
    :return: perturbed signal
    """
    return x+np.random.normal(scale=sigma,size = len(x))

def generate_signals(k,m,noise,npoints,dt=0.1):
    t=np.linspace(0,npoints*dt,num=npoints+1)#+1 is just to have round numbers
    sol = analytic_solution(k,m,t)
    meas = add_noise(sol,noise)
    return sol,meas

def KalmanFilterParameters(k,m,dt):
    F=np.array([[1,dt],[-k*dt/m,1]])


def RunKalmanFilter(measurements,kf):
    pass

if __name__ == '__main__':
    k=10
    m=3
    npoints=100
    noise=0.25
    dt=0.1
    true,meas = generate_signals(k,m,noise,npoints,dt)
    KalmanFilterParameters(k,m,dt)
    plt.figure()
    lines_true = plt.plot(true, color='b')
    dots_meas = plt.plot(meas, 'go', )
    plt.legend((lines_true[0], dots_meas[0]),
              ('true', 'measures'), loc='lower right')
    plt.show()