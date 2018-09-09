#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 20:12:52 2016 @author: Haibo Shi
"""
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
import tflearn
from numpy.linalg import inv,lstsq
from plant_dynamics import plantDynamics
from replay_buffer import ReplayBuffer


kf=25
dt=0.1


dimState = 4
dimContext = 4
dimAction = 2
centerTheta = np.array([np.pi*2/12,np.pi*7/12,0,0])
targetTheta = np.array([0.2099,1.5974,0,0])
states = {'center':centerTheta,'target':targetTheta}
buffer_size = 1e5
batch_size = 128;
learning_rate = 1e-5;

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

  # Create the model
#x = tf.placeholder(tf.float32, [None, 8])
#W0 = weight_variable([8, 100])
#b0 = bias_variable([100])
#W1 = weight_variable([100, 50])
#b1 = bias_variable([50])
#W2 = weight_variable([50, 2])
#b2 = tf.Variable(tf.zeros([2]))
#
#net0 = tf.nn.relu(tf.matmul(x,W0)+b0)
#net1 = tf.nn.relu(tf.matmul(net0,W1)+b1)
#y = tf.nn.tanh(tf.matmul(net1, W2) + b2)

x = tflearn.input_data(shape=[None, 8])
net = tflearn.fully_connected(x, 400, activation='relu')
net = tflearn.fully_connected(net, 300, activation='relu')
# Final layer weights are init to Uniform[-3e-3, 3e-3]
w_init = tflearn.initializations.uniform(minval=-0.0003, maxval=0.0003)
out = tflearn.fully_connected(net, 2, activation='tanh', weights_init=w_init)
y = tf.mul(out, 0.1) # Scale output to -action_bound to action_bound


action_gradient = tf.placeholder(tf.float32, [None, 2])
network_params = tf.trainable_variables()

actor_gradients = tf.gradients(y, network_params, -action_gradient)
train_step = tf.train.AdamOptimizer(learning_rate).\
            apply_gradients(zip(actor_gradients, network_params))

            
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())



dXfdX = 1e-1*np.zeros([kf,dimState,dimState])
dXfdU = 1e-1*np.zeros([kf,dimState,dimAction])
replay_buffer = ReplayBuffer(buffer_size)

observations, inlets, dXdotdX, dXdotdU =[],[],[],[]
plant = plantDynamics('center_out',0,states) #'random' for generating random points, 'fixed' for th eexample start and target points
for episode in range(10000):    
    observation = plant.reset()
    context = plant.taskCfg['states']['target']
    for k in range(kf):
        inlet = np.concatenate([observation,context])
        inlets.append(inlet)
        action = sess.run(y,feed_dict={x:np.reshape(inlet, (1, dimState+dimContext))})
        
        new_observation, error, done, info = plant.step(action[0])
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
        replay_buffer.add(np.reshape(inlets[k], (dimState+dimContext,)), np.reshape(grad_a, (dimAction,)))
    print ' | Episode', episode, '| error: ', np.dot(error,error)
    
#        if episode > 200:
    s_batch, grad_a_batch = replay_buffer.sample_batch(batch_size)
    sess.run(train_step,feed_dict={x:s_batch,action_gradient:grad_a_batch})

    replay_buffer.clear
    observations, inlets,dXdotdX, dXdotdU = [],[],[],[]


#%% validation

states_episodes = []
plt.hold(True)
for episode in range(16):    
    observation = plant.reset()
    context = plant.taskCfg['states']['target']
    states_rec = []
    for k in range(kf):
        states_rec.append(observation)
        inlet = np.concatenate([observation,context])
        action = sess.run(y,feed_dict={x:np.reshape(inlet, (1, dimState+dimContext))})
        
        new_observation, error, done, info = plant.step(action[0])
        observations.append(observation)        
        observation = new_observation 
        plt.plot(plant.x2y(observation[0:2])[0],plant.x2y(observation[0:2])[1],'o')
    