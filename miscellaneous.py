import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from copy import copy

def rescale_noise(pks,max_val):
    peaks = []
    for m in range(len(pks)):
        pk = pks[m]
        pk = pk/pk.max()
        pk *= max_val



        pk = np.random.poisson(pk)


        pk[pk<0] = 0
          
        pk[ pk==0] = np.sort(np.unique(pk.ravel()))[1]

        peaks.append(pk)
    return peaks


def center_u(u,support):
    support = support
    for i in range(3):
        t = copy(u[i])
        t[support] -= t[support].mean()
        u[i] = t
    return u

def calc_strain(u,step_size):
    u = tf.Variable(u)
#     step_size = tf.Variable(step_size,dtype=tf.float32)
    ux_x,ux_y,ux_z = tf.math.add(u[0,2:,:-2,:-2], -u[0,:-2,:-2,:-2])/2,tf.math.add(u[0,:-2,2:,:-2], -u[0,:-2,:-2,:-2])/2,tf.math.add(u[0,:-2,:-2,2:], -u[0,:-2,:-2,:-2])/2
    uy_x,uy_y,uy_z = tf.math.add(u[1,2:,:-2,:-2], -u[1,:-2,:-2,:-2])/2,tf.math.add(u[1,:-2,2:,:-2], -u[1,:-2,:-2,:-2])/2,tf.math.add(u[1,:-2,:-2,2:], -u[1,:-2,:-2,:-2])/2
    uz_x,uz_y,uz_z = tf.math.add(u[2,2:,:-2,:-2], -u[2,:-2,:-2,:-2])/2,tf.math.add(u[2,:-2,2:,:-2], -u[2,:-2,:-2,:-2])/2,tf.math.add(u[2,:-2,:-2,2:], -u[2,:-2,:-2,:-2])/2


    exx = ux_x
    eyy = uy_y
    ezz = uz_z
    exy = 0.5*tf.math.add(ux_y,uy_x)
    exz = 0.5*tf.math.add(ux_z,uz_x)
    eyz = 0.5*tf.math.add(uy_z,uz_y)
    
    strain = tf.stack([exx,eyy,ezz,exy,exz,eyz])/step_size
    strain = tf.pad(strain,[[0,0],[0,2],[0,2],[0,2]])

    return strain

def plot_strain(strain,min_max=False):
    fig,axs = plt.subplots(nrows=2,ncols=3,figsize=(15,7))
    c = np.array(strain.shape)[1]//2
    for i,ax in enumerate(axs.ravel()):
        if min_max != False:
            minn,maxx = min_max
        else:
            minn,maxx = strain[:,c-5:c+5,c-5:c+5,c-5:c+5].min(),strain[:,c-5:c+5,c-5:c+5,c-5:c+5].max()
        A = ax.imshow(strain[i,:,:,c],vmin=minn,vmax=maxx)
        plt.colorbar(A,ax=ax)
    plt.show()
    return
def plot_u(u,minmax = None,axis=2):
    fig,axs = plt.subplots(nrows=1,ncols=3,figsize= (15,5))
    c = np.array(u.shape)//2
    labels = ['$u_x$','$u_y$','$u_z$']
    for i,ax in enumerate(axs):
        s = u[i,c[1]-5:c[1]+5,c[2]-5:c[2]+5,c[3]-5:c[3]+5]   
        if minmax == None:
            if axis == 0:
                
                A = ax.imshow(u[i,c[1],:,:])
            if axis == 1:
                A = ax.imshow(u[i,:,c[2],:])
            if axis == 2:
                A = ax.imshow(u[i,:,:,c[3]])
        else:
            if axis == 0:
                A = ax.imshow(u[i,c[1],:,:],vmin=minmax[0],vmax=minmax[1])
            if axis == 1:
                A = ax.imshow(u[i,:,c[2],:],vmin=minmax[0],vmax=minmax[1])
            if axis == 2:
                A = ax.imshow(u[i,:,:,c[3]],vmin=minmax[0],vmax=minmax[1])
            
        ax.set_title(labels[i])
        fig.colorbar(A,ax=ax)
    plt.show()
    return
def translate(obj,loc1,loc2):
    shift = np.array(loc2)-np.array(loc1)
    obj = np.roll(obj,shift,axis=[-3,-2,-1])
    return obj

def pad(obj,df):
    if len(list(obj.shape)) == 3:
        obj = np.pad(obj,[(df[0],df[0]),(df[1],df[1]),(df[2],df[2])])
    else:
        obj = np.pad(obj,[(0,0),(df[0],df[0]),(df[1],df[1]),(df[2],df[2])])
    return obj
