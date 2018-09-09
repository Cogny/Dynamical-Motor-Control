#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 17:46:21 2017 @author: Haibo Shi
"""
import numpy as np
import tensorflow as tf
from plant_dynamics import plantDynamics
from replay_buffer_episodic import ReplayBuffer
kf=25
dt=0.1
dimState = 4
dimContext = 4
dimAction = 2
centerTheta = np.array([np.pi*2/12,np.pi*7/12,0,0])
targetTheta = np.array([0.2099,1.5974,0,0])
arm_states = {'center':centerTheta,'target':targetTheta}
buffer_size = 52
batch_size = 1;
learning_rate = 5e-5;
num_hidden = 64
error_across_episodes_total=[]   
# Create the model
input_data = tf.placeholder(tf.float32, [batch_size, kf,8])
input_single = tf.placeholder(tf.float32, [batch_size, 8])
state_single_input = tf.placeholder(tf.float32, [batch_size, num_hidden])
action_gradient = tf.placeholder(tf.float32, [batch_size, kf,dimAction])
cell = tf.nn.rnn_cell.BasicRNNCell(num_hidden)#cell = tf.nn.rnn_cell.BasicRNNCell(num_hidden)
#state_single = tf.Variable(tf.zeros_initializer([batch_size,num_hidden]), trainable= False)
#state_single = tf.get_variable('state_reset', [batch_size,num_hidden],\
#                               initializer = tf.constant_initializer(0.0), trainable = False)
with tf.variable_scope("M1") as RNN_scope:
    output_single, state_single_output = cell(input_single, state_single_input)    
    RNN_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=RNN_scope.name)
cell_outputs = []
cell_state = cell.zero_state(batch_size, tf.float32)
with tf.variable_scope("M1",reuse = True):
    for time_step in range(kf):
        (cell_output, cell_state) = cell(input_data[:, time_step, :], cell_state)
        cell_outputs.append(cell_output)
    outputs_pack = tf.pack(cell_outputs)
    outputs_transpose = tf.transpose(outputs_pack,[1,0,2])
        #outputs, states = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
weight = tf.Variable(tf.truncated_normal([num_hidden, dimAction])/num_hidden)
bias = tf.Variable(tf.constant(0.1, shape=[dimAction]))
final_projection = lambda x: tf.nn.tanh(tf.matmul(x, weight) + bias)

y_single = final_projection(output_single)
y_seq = tf.map_fn(final_projection, outputs_transpose) #batch_multiplicate

network_params = tf.trainable_variables()
lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in network_params]) * 0.008
reg_gradients = tf.gradients(lossL2,network_params)
actor_gradients = tf.gradients(y_seq[0], network_params, -action_gradient[0])
total_gradients = [a_+r_ for a_,r_ in zip(actor_gradients,reg_gradients)]
train_batch = tf.train.AdamOptimizer(learning_rate).\
            apply_gradients(zip(total_gradients, network_params)) 
projection_params = [weight,bias]
actor_gradients_projonly = tf.gradients(y_seq[0], projection_params, -2e-3*y_seq[0]-action_gradient[0])
train_batch_projonly = tf.train.AdamOptimizer(learning_rate).\
            apply_gradients(zip(actor_gradients_projonly, projection_params))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
dXfdX = 1e-1*np.zeros([kf,dimState,dimState])
dXfdU = 1e-1*np.zeros([kf,dimState,dimAction])
episode_x = np.zeros([kf,dimState+dimContext])
episode_grad_a = np.zeros([kf,dimAction])
x_batch = np.zeros([batch_size,kf,dimState+dimContext])
grad_a_batch = np.zeros([batch_size,kf,dimAction])
observations, inlets, dXdotdX, dXdotdU =[],[],[],[]
net_state = np.zeros([batch_size, num_hidden])
#%%
plant = plantDynamics('random_local',kf,arm_states) #'random' for generating random points, 'fixed' for th example start and target points
replay_buffer = ReplayBuffer(buffer_size)
error_across_episodes = []
for episode in range(np.int(50e3)):
    batch_ind = 0    
    observation = plant.reset()
    context = plant.taskCfg['states']['target']
    for k in range(kf):
        inlet = np.concatenate([observation,context])
        inlets.append(inlet)
        action, net_state = sess.run([y_single, state_single_output], \
                                     feed_dict={input_single:np.reshape(inlet, (1, dimState+dimContext)),\
                                                state_single_input: net_state})
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
        episode_x[k] =  np.reshape(inlets[k], (dimState+dimContext,))
        episode_grad_a[k] = np.reshape(grad_a, (dimAction,))
    x_batch[batch_ind] = episode_x
    grad_a_batch[batch_ind] = episode_grad_a 
    print ' | Episode', episode, '| error: ', np.dot(error,error)  
    net_state = np.zeros([batch_size, num_hidden])
#    error_across_episodes.append(np.dot(error,error))
    replay_buffer.add(np.vstack(observations),np.dot(error,error))
    s_episodic_batch, error_batch = replay_buffer.sample_batch(buffer_size)
    if np.mean(error_batch)<0.003:
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import matplotlib.cm as cmx
        e_m = np.mean(error_batch)
        e_s = np.std(error_batch)
        jet = plt.get_cmap('jet') 
        cNorm  = colors.Normalize(e_m-2*e_s, e_m+2*e_s)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        plt.hold(True)
        ax.set_xlim(-0,3,0.3)
        ax.set_ylim(0,1.,0.7)
        plt.grid(True)
        plt.axis('equal')
        for episode in range (buffer_size):
            colorVal = scalarMap.to_rgba(error_batch[episode])    
            for t in range(kf):
                endeffector = plant.x2y(s_episodic_batch[episode,t,0:2])
                plt.plot(endeffector[0],endeffector[1],'o',color=colorVal)
        break
    observations, inlets,dXdotdX, dXdotdU = [],[],[],[]
    batch_ind +=1
    if batch_ind==batch_size:
        batch_ind =0
        if episode<50e3:
            sess.run(train_batch,feed_dict={input_data:x_batch,action_gradient:grad_a_batch})
        else:
            sess.run(train_batch_projonly,feed_dict={input_data:x_batch,action_gradient:grad_a_batch})
##%%
#plant = plantDynamics('validation',kf,arm_states) #'random' for generating random points, 'fixed' for th example start and target points
#replay_buffer_2 = ReplayBuffer(buffer_size)
#for episode in range(np.int(16)):
#    batch_ind = 0    
#    observation = plant.reset()
#    context = plant.taskCfg['states']['target']
#    for k in range(kf):
#        inlet = np.concatenate([observation,context])
#        inlets.append(inlet)
#        action, net_state = sess.run([y_single, state_single_output], \
#                                     feed_dict={input_single:np.reshape(inlet, (1, dimState+dimContext)),\
#                                                state_single_input: net_state})
#        new_observation, error, done, info = plant.step(action[0])
#        observations.append(observation)        
#        
#        observation = new_observation       
#        dXdotdX.append(info['dXdotdX'])
#        dXdotdU.append(info['dXdotdU'])  
#    net_state = np.zeros([batch_size, num_hidden])
#    replay_buffer_2.add(np.vstack(observations),plant.ep_num)
#    observations, inlets,dXdotdX, dXdotdU = [],[],[],[]
#s_episodic_batch, ind_batch = replay_buffer_2.sample_batch(buffer_size)
#import matplotlib.pyplot as plt
#ax = plt.gca()
#import matplotlib.colors as colors
#import matplotlib.cm as cmx
#e_m = np.mean(ind_batch)
#e_s = np.std(ind_batch)
#jet = plt.get_cmap('jet') 
#cNorm  = colors.Normalize(e_m-2*e_s, e_m+2*e_s)
#scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
#plt.figure()
#plt.hold(True)
#for episode in range (16):
#    colorVal = scalarMap.to_rgba(ind_batch[episode])    
#    for t in range(kf):
#        endeffector = plant.x2y(s_episodic_batch[episode,t,0:2])
#        ax.set_xlim(-0,3,0.3)
#        ax.set_ylim(0,1.,0.7)
#        plt.grid(True)
#        plt.axis('equal')
#        plt.plot(endeffector[0],endeffector[1],'o',color=colorVal)

