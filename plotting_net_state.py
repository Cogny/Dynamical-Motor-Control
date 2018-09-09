#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 19:39:05 2017 @author: Haibo
"""
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import numpy as np
import tensorflow as tf
from plant_dynamics import plantDynamics
from replay_buffer_episodic import ReplayBuffer

states = np.load('net_state_rec_exp_list.npy')
pop_traj = states[7]

plt.figure()
plt.hold(True)
for d in range (32): 
        plt.plot(pop_traj[d,5:,1],'-b')