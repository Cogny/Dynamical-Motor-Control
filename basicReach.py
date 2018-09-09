# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 19:18:39 2016 @author: Haibo Shi
"""
import numpy as np
import time
from numpy.linalg import inv,lstsq
from plant_dynamics import plantDynamics
kf=25
dt=0.1
eta=1e-2
angRes=16
radius=0.19
dimState = 4
dimAction = 2
centerTheta = np.array([np.pi*2/12,np.pi*7/12,0,0])
targetTheta = np.array([0.2099,1.5974,0,0])
states = {'center':centerTheta,'end':targetTheta}
batch_size = 1;
damping = 1e-1;

u = 1e-3*np.ones([kf,dimAction])
dXfdX = 1e-1*np.zeros([kf,dimState,dimState])
dXfdU = 1e-1*np.zeros([kf,dimState,dimAction])
observations, dXdotdX, dXdotdU = [],[],[]
for trial in range(300):
    plant = plantDynamics('fixed',kf,states) #'random' for generating random points, 'fixed' for th eexample start and target points
    observation = plant.reset()
    for k in range(kf):
        action = u[k]
#        plant.render()
#        
#        time.sleep(1e-1)        
        new_observation, error, done, info = plant.step(action)
        observations.append(observation)        
        observation = new_observation        
        dXdotdX.append(info['dXdotdX'])
        dXdotdU.append(info['dXdotdU']) 
        
    # forward sweep
    for k in reversed(range(kf)):
        if k==kf-1:
            dXfdX[k]=dt*(dXdotdX[k])+np.eye(dimState)
            dXfdU[k]=dt*np.dot(dXfdX[k],dXdotdU[k])
        else:
            dXfdX[k]=dt*np.dot(dXfdX[k+1],dXdotdX[k])+dXfdX[k+1]
            dXfdU[k]=dt*np.dot(dXfdX[k+1],dXdotdU[k])
        
    print ' | Episode', trial, '| error: ', np.dot(error,error)
    # backward sweep
    grad_u = np.concatenate([_ for _ in dXfdU],axis=1)
    
    vecU=u.flatten()
#    vecU=vecU+np.dot(pinv(grad_u),error)
    cov = np.dot(grad_u.T,grad_u)+damping*np.eye(kf*dimAction)
    vecg = np.dot(grad_u.T,error)
    vecU = vecU + lstsq(cov,vecg)[0]
    u = vecU.reshape(kf,dimAction)
    observations, dXdotdX, dXdotdU = [],[],[]
    