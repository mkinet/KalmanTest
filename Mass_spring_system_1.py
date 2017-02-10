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

def InitKalmanFilter(k,m,dt,x0,sig0,noise):
    transition_matrix=np.array([[1,dt],[-k*dt/m,1]]) #state transition matrix
    observation_matrix=np.array([[1],[0]]).T # observation matrix
    initial_transition_covariance = np.eye(2)*1e-1
    initial_observation_covariance = np.array([0.5**2])
    transition_offsets=np.zeros((2,1))
    observation_offset=np.zeros(1)
    initial_state_mean=np.array([x0,0]) # initial estimation
    initial_state_covariance= np.eye(2)*sig0 # initial state variance
    kf = KalmanFilter(transition_matrices=transition_matrix,
                      observation_matrices=observation_matrix,
                      transition_covariance=initial_transition_covariance,
                      observation_covariance=initial_observation_covariance,
                      initial_state_mean=initial_state_mean,
                      initial_state_covariance=initial_state_covariance
                      )

    return kf

def RunKalmanFilter(nsteps,ndim,measurements,kf):
    filtered_state_means = np.zeros((nsteps, ndim))
    filtered_state_covariances = np.zeros((nsteps, ndim, ndim))
    for t in range(nsteps - 1):
        if t == 0:
            filtered_state_means[t] = kf.initial_state_mean
            filtered_state_covariances[t] = kf.initial_state_covariance
        filtered_state_means[t + 1], filtered_state_covariances[t + 1] = (
            kf.filter_update(filtered_state_means[t],
                             filtered_state_covariances[t],
                             measurements[t + 1]
                             )
            )

    return filtered_state_means

if __name__ == '__main__':
    k=5
    m=10
    nsteps=1000
    noise=0.3
    dt=0.5
    true,meas = generate_signals(k, m, noise, nsteps, dt)
    kf = InitKalmanFilter(k,m,dt,x0=1.0,sig0=0.1,noise=noise)
    est = RunKalmanFilter(nsteps,ndim=2,measurements=meas,kf=kf)
    plt.figure()
    lines_true = plt.plot(true, color='b')
    lines_filter = plt.plot(est[:,0],color = 'r')
    dots_meas = plt.plot(meas, 'go', )
    plt.legend((lines_true[0], lines_filter[0], dots_meas[0]),
              ('true', 'estimate', 'measures'), loc='lower right')
    plt.show()