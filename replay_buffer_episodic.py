#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 21:55:06 2017 @author: Haibo Shi
"""
from collections import deque
import random
import numpy as np

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences 
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
#        random.seed(random_seed)

    def add(self, s_episode, error):
        experience = (s_episode, error)
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_episode_batch = np.array([_[0] for _ in batch])
        error_batch = np.array([_[1] for _ in batch])


        return s_episode_batch, error_batch

    def clear(self):
        self.deque.clear()
        self.count = 0



