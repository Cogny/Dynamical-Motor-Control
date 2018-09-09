# -*- coding: utf-8 -*-
"""
plant dynamics of a non-liner arm
Created on Mon Nov 14 15:45:42 2016 @author: Haibo Shi
"""
import numpy as np
from numpy.linalg import inv,pinv,norm
#import matplotlib.pyplot as plt


class plantDynamics():
    def __init__(self,pointsGen_mode,curlInd,statesfromto):    
        self.ep_num = 0
        self.taskCfg = dict(dt=0.1,mode = pointsGen_mode,episode_length=25,states=statesfromto)
        self.states_fixed = statesfromto
        self.physics = dict(b11=0.05,b22=0.05,b12=0,b21=0,m1=1.4,m2=1,\
        l1=0.3,l2=0.33,s1=0.11,s2=0.16,I1=0.025,I2=0.045)
        self.physics['a1']=self.physics['I1']+self.physics['I2']\
        +self.physics['m2']*self.physics['l1']**2
        self.physics['a2']=self.physics['m2']*self.physics['l1']*self.physics['s2']
        self.physics['a3']=self.physics['I2']
        if curlInd ==0:
            self.physics['b12']=self.physics['b21']=0
        elif curlInd==-1:
            self.physics['b12']=-1.
            self.physics['b21']=1.
        elif curlInd==1:
            self.physics['b12']=0.75 #1.5
            self.physics['b21']=-0.75 #1.5
#    def pointsGen(self):
#        distance_stroke = 0.2
#        l02 = self.physics['l1']+self.physics['l2']
#        xlim,ylim = np.array([-np.sqrt(.5)*l02,np.sqrt(.5)*l02]),np.array([-np.sqrt(.5)*l02,np.sqrt(.5)*l02])
#        xlim,ylim = 1*xlim,1*ylim
#        theta_start_rand = np.random.uniform(-np.pi,np.pi,[2,])
#        coor_start_rand = self.x2y(theta_start_rand)
#        while True:    
#            theta_stroke =  np.random.uniform(-np.pi,np.pi)
#            coor_end_rand = coor_start_rand + distance_stroke*np.array([np.cos(theta_stroke),np.sin(theta_stroke)])
#            if (coor_end_rand[0]>xlim[0] and coor_end_rand[0]<xlim[1])and(coor_end_rand[1]>ylim[0] and coor_end_rand[1]<ylim[1]):
#                break
#        theta_end_rand = theta_start_rand
#        for _ in range(20):
#            coor_temp_rand = self.x2y(theta_end_rand)
#            coor_error = coor_end_rand-coor_temp_rand
#            theta_end_rand = theta_end_rand + 3e-1*np.dot(pinv(self.jacobian(theta_end_rand)),coor_error)
##        theta_end_rand[0] = np.angle([np.cos(theta_end_rand[0])+np.sin(theta_end_rand[0])*1j])
##        theta_end_rand[1] = np.angle([np.cos(theta_end_rand[1])+np.sin(theta_end_rand[1])*1j])    
#        return {'start':theta_start_rand,"target":theta_end_rand,'coor_end_rand':coor_end_rand}
    def pointsGen_local(self):
        theta_center_local = self.states_fixed['center'][0:2]
        coor_center_local = self.x2y(theta_center_local)
        radius_local = 0.25
        distance_local = np.random.uniform(0.05,radius_local)
        theta_local = np.random.uniform(-np.pi,np.pi)
        coor_start_rand = coor_center_local + distance_local*np.array([np.cos(theta_local),np.sin(theta_local)])
        theta_start_rand = theta_center_local
        for _ in range(20):
            coor_temp_rand = self.x2y(theta_start_rand)
            coor_error = coor_start_rand-coor_temp_rand
            theta_start_rand = theta_start_rand + 3e-1*np.dot(pinv(self.jacobian(theta_start_rand)),coor_error)
        theta_start_rand[0] = np.angle([np.cos(theta_start_rand[0])+np.sin(theta_start_rand[0])*1j])
        theta_start_rand[1] = np.angle([np.cos(theta_start_rand[1])+np.sin(theta_start_rand[1])*1j])    
        distance_stroke = 0.2
        while True:    
            theta_stroke =  np.random.uniform(-np.pi,np.pi)
            coor_end_rand = coor_start_rand + distance_stroke*np.array([np.cos(theta_stroke),np.sin(theta_stroke)])
            if (norm(coor_end_rand-coor_center_local) <= radius_local):
                break
        theta_end_rand = theta_start_rand
        for _ in range(20):
            coor_temp_rand = self.x2y(theta_end_rand)
            coor_error = coor_end_rand - coor_temp_rand
            theta_end_rand = theta_end_rand + 3e-1*np.dot(pinv(self.jacobian(theta_end_rand)),coor_error)
        theta_end_rand[0] = np.angle([np.cos(theta_end_rand[0])+np.sin(theta_end_rand[0])*1j])
        theta_end_rand[1] = np.angle([np.cos(theta_end_rand[1])+np.sin(theta_end_rand[1])*1j])    
        return {'start':theta_start_rand,"target":theta_end_rand,'coor_end_rand':coor_end_rand}

    def pointsGen_center_out(self):
        theta_center_local = self.states_fixed['center'][0:2]
        coor_center_local = self.x2y(theta_center_local)
        radius_local = 0.2
        theta_local = np.random.uniform(-np.pi,np.pi)
        coor_radial_rand = coor_center_local + radius_local*np.array([np.cos(theta_local),np.sin(theta_local)])
        theta_radial_rand = theta_center_local
        for _ in range(20):
            coor_temp_rand = self.x2y(theta_radial_rand)
            coor_error = coor_radial_rand-coor_temp_rand
            theta_radial_rand = theta_radial_rand + 3e-1*np.dot(pinv(self.jacobian(theta_radial_rand)),coor_error)
        theta_radial_rand[0] = np.angle([np.cos(theta_radial_rand[0])+np.sin(theta_radial_rand[0])*1j])
        theta_radial_rand[1] = np.angle([np.cos(theta_radial_rand[1])+np.sin(theta_radial_rand[1])*1j])    
        return {'start':theta_center_local,"target":theta_radial_rand,'coor_end_rand':coor_radial_rand}
    
    def pointsGen_validation(self):
        theta_center_local = self.states_fixed['center'][0:2]
        coor_center_local = self.x2y(theta_center_local)
        radius_local = 0.2
        theta_local = (self.ep_num-1)*np.pi/8
        coor_radial_rand = coor_center_local + radius_local*np.array([np.cos(theta_local),np.sin(theta_local)])
        theta_radial_rand = theta_center_local
        for _ in range(20):
            coor_temp_rand = self.x2y(theta_radial_rand)
            coor_error = coor_radial_rand-coor_temp_rand
            theta_radial_rand = theta_radial_rand + 3e-1*np.dot(pinv(self.jacobian(theta_radial_rand)),coor_error)
        theta_radial_rand[0] = np.angle([np.cos(theta_radial_rand[0])+np.sin(theta_radial_rand[0])*1j])
        theta_radial_rand[1] = np.angle([np.cos(theta_radial_rand[1])+np.sin(theta_radial_rand[1])*1j])    
        return {'start':theta_center_local,"target":theta_radial_rand,'coor_end_rand':coor_radial_rand}
    def x2y(self,theta):
        return np.array([self.physics['l1']*np.cos(theta[0])+self.physics['l2']*np.cos(theta[0]+theta[1]),\
        self.physics['l1']*np.sin(theta[0])+self.physics['l2']*np.sin(theta[0]+theta[1])])
    def jacobian(self,theta):
        return np.array([[-self.physics['l1']*np.sin(theta[0])-self.physics['l2']*np.sin(theta[0]+theta[1]),\
                          -self.physics['l2']*np.sin(theta[0]+theta[1])],\
                         [self.physics['l1']*np.cos(theta[0])+self.physics['l2']*np.cos(theta[0]+theta[1]),\
                          self.physics['l2']*np.cos(theta[0]+theta[1])]])
    def linearity(self,ob_seq):
        coor_seq = np.array([self.x2y(_[0:2]) for _ in ob_seq])
        corr = np.corrcoef(coor_seq[:,0],coor_seq[:,1])[0,1]
        return np.abs(corr)
            
        
        
        
        
        
        
    def reset(self):
        self.ep_num = self.ep_num+1
        self.time_step=0                
        if self.taskCfg['mode'] == 'center_out':
            generated_points = self.pointsGen_center_out()
        elif self.taskCfg['mode'] == 'random_local':
            generated_points = self.pointsGen_local()
        elif self.taskCfg['mode'] == 'random_global':
            generated_points = self.pointsGen()
        elif self.taskCfg['mode'] == 'validation':
            generated_points = self.pointsGen_validation()
        elif self.taskCfg['mode'] == 'fixed':
            generated_points ={'start':self.states_fixed['center'][0:2],'target': self.states_fixed['end'][0:2]}
        self.taskCfg['states']['start']=np.concatenate([generated_points['start'],np.zeros([2,])])
        self.taskCfg['states']['target']=np.concatenate([generated_points['target'],np.zeros([2,])])
        self.ob=self.taskCfg['states']['start']
        self.done = False
        self.rew = 0.
        self.error = np.zeros([4,])
        return self.ob
    def step(self,action):
        self.time_step += 1
        
        M=np.array([[self.physics['a1']+2*self.physics['a2']*np.cos(self.ob[1]),\
        self.physics['a3']+self.physics['a2']*np.cos(self.ob[1])],\
        [self.physics['a3']+self.physics['a2']*np.cos(self.ob[1]),\
        self.physics['a3']]])
        a2sinx=self.physics['a2']*np.sin(self.ob[1])
        C=np.array([a2sinx*(-1*self.ob[3]*(2*self.ob[2]+self.ob[3])),\
        a2sinx*self.ob[2]**2]).T
        B=np.array([[self.physics['b11'],self.physics['b12']],\
        [self.physics['b21'], self.physics['b22']]])
        B=0.1*B
        MInverse=inv(M)
        F=action-C-np.dot(B,self.ob[2:4])
        A=np.dot(MInverse,F)
        obDot = np.concatenate([self.ob[2:4],A])
        #differential computation
        dCdX=np.array([[0,\
        (-self.physics['a2']*self.ob[3]*(2*self.ob[2]+self.ob[3])*np.cos(self.ob[1])),\
        (-2*self.physics['a2']*np.sin(self.ob[1])*self.ob[3]),\
        (-(2*self.ob[2]+2*self.ob[3])*self.physics['a2']*np.sin(self.ob[1]))],\
        [0,\
        self.physics['a2']*((self.ob[2])**2)*np.cos(self.ob[1]),\
        2*self.physics['a2']*self.ob[2]*np.sin(self.ob[1]),\
        0]])
        BdAngVdX=np.dot(B,np.array([[0,0,1,0],[0,0,0,1]]))
        
        dFdX=-dCdX-BdAngVdX
        dMdXA=np.array([[0,\
        -(2*A[0]*self.physics['a2']*np.sin(self.ob[1])+A[1]*self.physics['a2']*np.sin(self.ob[1])),0,0],\
        [0,-(A[0]*self.physics['a2']*np.sin(self.ob[1])),0,0]])
        dAdX=np.dot(MInverse,(dFdX-dMdXA))
        shift = np.concatenate([np.zeros([2,2]),np.eye(2)],axis=1)
        dXdotdX=np.concatenate([shift,dAdX],axis=0)
        dFdU=np.eye(2)
        dAdU=np.dot(MInverse,dFdU)
        dXdotdU=np.concatenate([np.zeros([2,2]),dAdU],axis=0)
        self.ob = self.ob +self.taskCfg['dt']*obDot
        if self.time_step == self.taskCfg['episode_length']:
            self.done = True
            self.rew = norm(self.ob-self.taskCfg['states']['target'])
            z_end_0 = np.cos(self.taskCfg['states']['target'][0])+np.sin(self.taskCfg['states']['target'][0])*1j
            z_end_1 = np.cos(self.taskCfg['states']['target'][1])+np.sin(self.taskCfg['states']['target'][1])*1j
            z_ob_0 = np.cos(self.ob[0])+np.sin(self.ob[0])*1j
            z_ob_1 = np.cos(self.ob[1])+np.sin(self.ob[1])*1j
            self.error[0] = np.angle(z_end_0/z_ob_0)
            self.error[1] = np.angle(z_end_1/z_ob_1)
            self.error[2:4] = self.taskCfg['states']['target'][2:4]- self.ob[2:4]
#            self.error = self.taskCfg['states']['target']-self.ob
        return self.ob, self.error, self.done, {'dXdotdX':dXdotdX,'dXdotdU':dXdotdU}
#    def render(self):
#        plt.subplot(221)
#        plt.plot(np.array([0.,self.physics['l1']*np.cos(self.ob[0])]),\
#        np.array([0.,self.physics['l1']*np.sin(self.ob[0])]),'*-')
#        plt.axis([-0.7,0.7,-0.7,0.7])             
#        plt.hold(True)
#        plt.plot(self.physics['l1']*np.cos(self.taskCfg['states']['target'][0])+\
#        self.physics['l2']*np.cos(self.taskCfg['states']['target'][0]+self.taskCfg['states']['target'][1]),\
#        self.physics['l1']*np.sin(self.taskCfg['states']['target'][0])+\
#        self.physics['l2']*np.sin(self.taskCfg['states']['target'][0]+self.taskCfg['states']['target'][1]),'r*')
#        plt.plot(self.physics['l1']*np.cos(self.ob[0])+self.physics['l2']*np.cos(self.ob[0]+self.ob[1]),\
#        self.physics['l1']*np.sin(self.ob[0])+self.physics['l2']*np.sin(self.ob[0]+self.ob[1]),'*')
#        plt.plot(np.array([self.physics['l1']*np.cos(self.ob[0]),\
#        self.physics['l1']*np.cos(self.ob[0])+self.physics['l2']*np.cos(self.ob[0]+self.ob[1])]),\
#        np.array([self.physics['l1']*np.sin(self.ob[0]),self.physics['l1']*np.sin(self.ob[0])+\
#        self.physics['l2']*np.sin(self.ob[0]+self.ob[1])]),'*-');
#        plt.axis([-0.7,0.7,-0.7,0.7])             
#        plt.hold(False)