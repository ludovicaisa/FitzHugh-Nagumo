#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 17:04:59 2023

@author: ludovica
"""
#%%
import numpy as np
import scipy.integrate
import scipy
#import sdeint
import matplotlib.pyplot as plt

#setting parameters for the simulation
T = 1000 # total time
dt = 0.01  # time step
a=1.2 # set a
eps=0.01 # set time scale parameter
I=0.01 #input current (constant)
v0=-0.8 #initial condition v
w0=-0.6 #initial condition w

p=(a,eps,I)
t=np.arange(0,T,0.01)
N = int(T/dt)
y0= [v0, w0]

def fitzhugh_nagumo(state, t, a, eps, I):
    """Differential equations [v,w] of the Fitzhugh-Nagumo system.      
    Return: dx/dt (array size 2)
    """
    v,w=state
    dv = (v - (v**3)/3 -w +I)/eps
    dw = (v+a)
    x = [dv,dw]
    return x

def isoclines(a,I):
  ''' Plot the nullclines in the phase space. Set parameter (a,I) '''
  v= np.linspace(-2.5,2.5,1000)
  c = v - (v**3)/3 +I
  plt.plot(v, c, label='v isocline', color= 'green')
  plt.plot([-a,-a],[3,-3], label='w isocline', color='orange')
  

def fixed_point_stab(a,I,eps):
    '''
     Returns FIXED POINT coordinate & Eigenvalues of the Jacobian Matrix of the system calculated on the fixed point
     to define the stability of the fixed point, with values (a,I, eps)
    '''
    v_eq=-a
    w_eq=v_eq-(v_eq**3)/3 +I
    p_eq=[v_eq,w_eq]
    print('fixed point=',p_eq)
    jacobian_matrix= np.array([[1-p_eq[0]**2, -1], [eps, 0]])
    np.linalg.eig(jacobian_matrix)
    eig_values=np.linalg.eig(jacobian_matrix)[0]
    print(eig_values)
    return[p_eq,eig_values]


  


def deterministic_traj(a,I):
    # integrate the trajectory through scipy method
    traj = scipy.integrate.odeint(fitzhugh_nagumo, y0, t, args=p)
    # Extract x[0] and x[1] from the trajectory
    x0_values = traj[:, 0]
    x1_values = traj[:, 1]
    #find the fixed point and eigenvalues of J(f) the system
    p_eq, eig = fixed_point_stab(a, I, eps)
    
    #Plot the phase space the variables with dt=0.01
    plt.figure()
    isoclines(a,I)
    plt.plot(x0_values, x1_values, 'x',  label='trajectory', color='blue' )
    plt.plot(p_eq[0], p_eq[1], 'o', label= 'fixed point', color= 'red')
    plt.plot(v0, w0, 'd', label= 'i. c.', color= 'black')
    plt.xlabel('v')
    plt.ylabel('w')
   #x, y limit can be changed at your willing
    plt.ylim(-1.5,1.5)
    plt.xlim(-2.5,2.5)
    
    plt.legend()
    plt.title('FHN: a={}, eps={}, I={}'.format(a,eps,I))
    plt.grid(linestyle='--')
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    plt.show()
    
    #Plot the time series of [v,w]
    plt.figure()
    plt.title('FHN: a={}, eps={}, I={}'.format(a,eps,I))
    plt.plot(t,x0_values, label='trajectory v')
    plt.plot(t,x1_values, label='trajectory w')
    plt.xlabel('time')
    plt.ylabel('v,w')
    plt.grid(linestyle='--')
    plt.legend()
    return [x0_values, x1_values, t]



def wiener(gamma, temp):
    ''' 
    Stochastic FHN with Wienernoise: it calculates the trajectory of [v,w,t]
    set
    gamma: dissipative variable
    temp: amplitude of the noise
    '''
# Initialization of arrays
    v = np.zeros(N)
    w = np.zeros(N)
# initial condition
    v[0] = v0
    w[0] = w0

# random gaussian noise added at each time step
    xi = np.random.normal(0, 1, N)

# Numerical integration with the Euler-Marujama method
    for i in range(1, N):
        v[i] = v[i-1] + dt * (v[i-1] - (v[i-1]**3)/3 - w[i-1] + I) 
        w[i] = w[i-1] + dt * eps * (v[i-1] + a-gamma*w[i-1]) + np.sqrt(2*temp*gamma) * xi[i]*np.sqrt(dt)*eps

# Saving trajectories
    t = np.linspace(0, T, N)
    
#Plot trajectories in the phase space
    plt.figure()
    plt.plot(v, w,'x', markersize=0.8, label='trajectory')
    plt.xlabel('v')
    plt.ylabel('w')
    isoclines(a,I)
    plt.ylim(-1,1.5)
    plt.xlim(-2.5,2.5)
    plt.legend()
    plt.title('Wiener ps,a={},temp={}, g={}'.format(a,temp, 0.5))
    plt.grid(linestyle='--')
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    plt.legend()

#Plot time series
    plt.figure()
    plt.plot(t,v,label='v')
    plt.plot(t,w, label= 'w')
    plt.xlabel('Time')
    plt.legend()
    plt.grid(linestyle='--')
    plt.ylabel('v,w')
    plt.title('Wiener time, temp={}, g={}'.format(temp, 0.5))
    plt.show()
    
    #Returns trajectory
    trajectory = [v,w,t]
    return trajectory




def number_spike(traj):
    ''' Count the number of spikes in the trajectory
    the threshold is set up as zero.  '''
    count=0
    x=False
    for i in range(0, N):
        if traj[0][i-1]<0 and traj[0][i]>= 0.0 and x==False:
            count +=1
            x=True
        if traj[0][i]<= 0.0 and traj[0][i-1]>=0.0 and x==True:
            x=False
    return count

def time_spike(traj):
    ''' Collect the time when spikes starts for all the spikes of the trajectory in the array timeisi.    
    the threshold is set up as zero for the v variable.  '''
    timeisi=[]
    x=False
    for i in range(0, N):
        if traj[0][i-1]<0 and traj[0][i]>= 0.0 and x==False:
            timeisi.append(traj[2][i])
            x=True
        if traj[0][i]<= 0.0 and traj[0][i-1]>=0.0 and x==True:
            x=False
    return timeisi

def isi_data(timeisi):
    ''' Count the time between a spike and previous one during the trajectory, (interspike time interval)
    and collects in the array data_isi. 
    '''
    data=[]
    for i in range(0,len(timeisi)-1):
        isi= timeisi[i+1]-timeisi[i]
        data.append(isi)
    data_isi = np.asarray(data)
    return data_isi

def est_intensity(data_isi):
    intensity= np.mean(data_isi)
    return intensity

def OU(gamma,temp):
    ''' 
    Stochastic FHN with Orstein-Uhnlenbeck noise: it calculates the trajectory of [v,w,t]
    set
    gamma: 'memory' variable
    temp: amplitude of the noise
    '''
    gamma=gamma
    temp=temp

# Initialization of arrays
    v_w = np.zeros(N)
    w_w = np.zeros(N)
    eta = np.zeros(N)

# Initial condition
    v_w[0] = v0
    w_w[0] = w0

# Generation of gaussian noise
    xi = np.random.normal(0, 1, N)

#Numerical integration with Euler-Marujama method
    for i in range(1, N):
        v_w[i] = v_w[i-1] + dt * (v_w[i-1] - (v_w[i-1]**3)/3 - w_w[i-1] + I)
        w_w[i] = w_w[i-1] + dt * eps* (v_w[i-1] + a + + eta[i-1])
        eta[i] = eta[i-1] - gamma * eta[i-1] * dt + temp * xi[i] * np.sqrt(dt)

# Time
    t = np.linspace(0, T, N)

#Plot in the Phase Space the trajectory
    plt.figure()
    plt.plot(v_w, w_w, 'x', markersize=0.8 ,label='w')
    isoclines(a,I)
    plt.ylim(-1,1.5)
    plt.xlim(-2.5,2.5)
    plt.xlabel('v')
    plt.ylabel('w')
    plt.legend()
    plt.title('OU ps: T={}, \u03B3={}'.format(temp,gamma))
    plt.grid(linestyle='--')
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    plt.legend()

#Plot the time series  of [v,w,n]   
    plt.figure()
    plt.plot(t,v_w,label='v')
    plt.plot(t,w_w, label= 'w')
    plt.xlabel('Time')
    plt.legend()
    plt.grid(linestyle='--')
    plt.ylabel('v,w')
    plt.title('OU time: temp={}, gamma={}'.format(temp,gamma))
    plt.show()
    
    plt.figure()
    plt.plot(t,eta, label='eta')
    plt.xlabel('time')
    plt.ylabel('eta')
    plt.show()
  
    trajectory=[v_w,w_w,eta, t]
    return trajectory
