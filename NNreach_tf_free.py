#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 20:12:52 2016 @author: Haibo Shi
"""

import numpy as np

from plant_dynamics import plantDynamics
from replay_buffer_fat import ReplayBuffer

kf=25
dt=0.1


dimState = 4
dimContext = 4
dimAction = 2
centerTheta = np.array([np.pi*2/12,np.pi*7/12,0,0])
targetTheta = np.array([0.2099,1.5974,0,0])
states = {'start':centerTheta,'target':targetTheta}
buffer_size = 1e5
batch_size = 64;
learning_rate = 1e-5;
decay_rate = 0.99

  # Create the model
model = {}
num_in = 9
num_h0 = 1000
num_h1 = 100
num_out = 2
model['W0'] = np.random.randn(num_in,num_h0) / np.sqrt(num_in)
model['W1'] = 0.1*np.random.randn(num_h0,num_h1) / np.sqrt(num_h0)
model['W2'] = 0.1*np.random.randn(num_h1,num_out) / np.sqrt(num_h1)

rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory

def policy_forward(x):
    h0 = np.dot(x,model['W0'])
    h0[h0<0] = 0 # ReLU nonlinearity
    h1 = np.dot(h0,model['W1'])
    h1[h1<0] = 0
    logit = np.dot(h1,model['W2'])
    act = np.tanh(logit)
    return act,h1,h0# return probability of taking action 2, and hidden state

def policy_backward(x_bat,h0_bat, h1_bat, action_bat, action_gradient_bat):
    """ backward pass. (eph is array of intermediate hidden states) """
    tanh_grad = (1.-action_bat)*(1.+action_bat)
    
    dlogit = tanh_grad*action_gradient_bat
    dW2 = np.dot(h1_bat.T,dlogit)
    dh1 = np.dot(dlogit, model['W2'].T)
    dh1[h1_bat <= 0] = 0 # backpro prelu
    dW1 = np.dot(h0_bat.T,dh1)
    dh0 = np.dot(dh1, model['W1'].T)
    dh0[h0_bat <= 0] = 0 # backpro prelu
    dW0 = np.dot(x_bat.T,dh0)
    return {'W0':dW0, 'W1':dW1,'W2':dW2}

dXfdX = 1e-1*np.zeros([kf,dimState,dimState])
dXfdU = 1e-1*np.zeros([kf,dimState,dimAction])
replay_buffer = ReplayBuffer(buffer_size)

observations,inlets, actions,h1s,h0s, dXdotdX, dXdotdU =[],[],[],[],[],[],[]
plant = plantDynamics('fixed',kf,states) #'random' for generating random points, 'fixed' for th eexample start and target points
for episode in range(5000):    
    observation = plant.reset()
    context = plant.taskCfg['states']['target']
    for k in range(kf):
        inlet = np.concatenate([observation,context,np.array([1.])])
        inlets.append(inlet)
        action,hid1,hid0 = policy_forward(inlet)
        actions.append(action)
        h1s.append(hid1)
        h0s.append(hid0)
        new_observation, error, done, info = plant.step(action)
        observations.append(observation)        
        observation = new_observation        
        dXdotdX.append(info['dXdotdX'])
        dXdotdU.append(info['dXdotdU']) 
        
    # backward sweep
    for k in reversed(range(kf)):
        if k==kf-1:
            dXfdX[k]=dt*(dXdotdX[k])+np.eye(dimState)
            dXfdU[k]=dt*np.dot(dXfdX[k],dXdotdU[k])
        else:
            dXfdX[k]=dt*np.dot(dXfdX[k+1],dXdotdX[k])+dXfdX[k+1]
            dXfdU[k]=dt*np.dot(dXfdX[k+1],dXdotdU[k])
        grad_a = np.dot(dXfdU[k].T,error)
        replay_buffer.add(inlets[k], h0s[k], h1s[k], actions[k], grad_a)
    print ' | Episode', episode, '| error: ', np.dot(error,error)
    
#        if episode > 200:
    x_batch, h0_batch, h1_batch, action_batch, action_gradient_batch = replay_buffer.sample_batch(batch_size)
    grad = policy_backward(x_batch, h0_batch, h1_batch, action_batch, action_gradient_batch)

    for name,v in model.iteritems():
        g = grad[name] # gradient
        rmsprop_cache[name] = decay_rate * rmsprop_cache[name] + (1 - decay_rate) * g**2
        model[name] += learning_rate * g / (np.sqrt(rmsprop_cache[name]) + 1e-5)

#    replay_buffer.clear
    observations,inlets, actions, h1s, h0s, dXdotdX, dXdotdU =[],[],[],[],[],[],[]
