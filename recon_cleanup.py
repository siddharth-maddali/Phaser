# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:26:41 2020

@author: mattw
"""
import numpy as np

from scipy.ndimage.measurements import center_of_mass
import matplotlib.pyplot as plt
import copy
import matplotlib.pyplot as plt
import itertools as it
import cv2

def check_mirrors(s):
    
    """ vals: input array that can be a list of [n,n,n] phase matrices or [3,n,n,n] u
            matrices, depending on the phase_or_u value"""

    #center them all

    axes = list(it.permutations(range(3)))
    flip = []
    centers = []
    for i in range(len(s)):
        s1 = s[i]
        error = np.sum((s[0]-s1)**2)
        
        
        # s1,val1 = center_object(s1,val1,phase_or_u = phase_or_u)
        s1_prime,center = center_object(np.flip(np.flip(np.flip(s1,axis=0),axis=1),axis=2))
        error1 = np.sum((s[0]-s1_prime)**2)
                
        if error1<error:
            print('flip')
            s[i] = s1_prime
            flip.append(True)
            centers.append(center)
        else:
            flip.append(False)
            centers.append(center)


    return s,flip,centers



def center_object(sup):
    center = np.array(np.round(center_of_mass(sup),0),dtype=np.int32) #center 
    for n in [ 0, 1, 2]:
        sup = np.roll(sup,sup.shape[n]//2 - center[n],axis=n)
    return sup,center

def plot_comparison(reconstructed,ground_truth,rng = 0.04,absolute=False):
    fig,ax = plt.subplots(nrows=2,ncols=3,figsize= (18,10))
    labels = ['$u_x$','$u_y$','$u_z$']
    for i in range(2):                           #plot side by side
        for j in range(3):
            if i == 0:
                A = ax[i,j].imshow(reconstructed[j,:,:,reconstructed.shape[3]//2],vmax = rng/2,vmin = -rng/2)
            else:
                A = ax[i,j].imshow(ground_truth[j,:,:,ground_truth.shape[3]//2],vmax = rng/2,vmin = -rng/2)
            ax[i,j].set_title(labels[j])
            t = fig.colorbar(A,ax=ax[i,j])
            t.set_label('nm')
    plt.show()

    diff_u = ground_truth-reconstructed               #plot difference between ground truth and reconstructed
    fig,ax = plt.subplots(ncols=3,figsize= (18,5))
    for i in range(3):
        A = ax[i].imshow(diff_u[i,:,:,diff_u.shape[3]//2],vmax = rng/2,vmin = -rng/2)
        ax[i].set_title('difference %s'%labels[i])
        b = fig.colorbar(A,ax=ax[i])
        b.set_label('nm')
    plt.show()
    if absolute:                       #plot absolute difference
        diff_u = np.absolute(ground_truth-reconstructed)
        fig,ax = plt.subplots(ncols=3,figsize= (18,5))
        for i in range(3):
            A = ax[i].imshow(diff_u[i,:,:,diff_u.shape[3]//2],vmax=rng,vmin = 0)
            ax[i].set_title('absolute difference %s'%labels[i])
            b = fig.colorbar(A,ax=ax[i])
            b.set_label('nm')
        plt.show()
        
    return

def erode(supp,kernel_shape=(3,3),it=1):
    sup = copy.copy(supp)
    for j in range(supp.shape[2]):
        kernel = np.ones(kernel_shape,np.uint8)
        sup[:,:,j] = cv2.erode(sup[:,:,j],kernel,iterations = it)
    return sup
