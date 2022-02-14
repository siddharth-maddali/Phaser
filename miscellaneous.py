import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from copy import copy
from spec import parse_spec

def fig_plot_u(u,minmax = None,axis=2):
    fig,axs = plt.subplots(nrows=1,ncols=3,figsize= (15,4))
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
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        cbar = fig.colorbar(A,ax=ax,shrink=0.9)

        cbar.set_label('nm', rotation=270)
    
    return fig
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

# def calc_strain(u,step_size):
#     u = tf.Variable(u)
# #     step_size = tf.Variable(step_size,dtype=tf.float32)
#     ux_x,ux_y,ux_z = tf.math.add(u[0,2:,:-2,:-2], -u[0,:-2,:-2,:-2])/2,tf.math.add(u[0,:-2,2:,:-2], -u[0,:-2,:-2,:-2])/2,tf.math.add(u[0,:-2,:-2,2:], -u[0,:-2,:-2,:-2])/2
#     uy_x,uy_y,uy_z = tf.math.add(u[1,2:,:-2,:-2], -u[1,:-2,:-2,:-2])/2,tf.math.add(u[1,:-2,2:,:-2], -u[1,:-2,:-2,:-2])/2,tf.math.add(u[1,:-2,:-2,2:], -u[1,:-2,:-2,:-2])/2
#     uz_x,uz_y,uz_z = tf.math.add(u[2,2:,:-2,:-2], -u[2,:-2,:-2,:-2])/2,tf.math.add(u[2,:-2,2:,:-2], -u[2,:-2,:-2,:-2])/2,tf.math.add(u[2,:-2,:-2,2:], -u[2,:-2,:-2,:-2])/2


#     exx = ux_x
#     eyy = uy_y
#     ezz = uz_z
#     exy = 0.5*tf.math.add(ux_y,uy_x)
#     exz = 0.5*tf.math.add(ux_z,uz_x)
#     eyz = 0.5*tf.math.add(uy_z,uz_y)
    
#     strain = tf.stack([exx,eyy,ezz,eyz,exz,exy])/step_size
#     strain = tf.pad(strain,[[0,0],[0,2],[0,2],[0,2]])

#     return strain
def calc_strain(u,step_size):
    
    ux_x,ux_y,ux_z = tf.math.add(u[0,1:,:-1,:-1], -u[0,:-1,:-1,:-1])/1,tf.math.add(u[0,:-1,1:,:-1], -u[0,:-1,:-1,:-1])/1,tf.math.add(u[0,:-1,:-1,1:], -u[0,:-1,:-1,:-1])/1
    uy_x,uy_y,uy_z = tf.math.add(u[1,1:,:-1,:-1], -u[1,:-1,:-1,:-1])/1,tf.math.add(u[1,:-1,1:,:-1], -u[1,:-1,:-1,:-1])/1,tf.math.add(u[1,:-1,:-1,1:], -u[1,:-1,:-1,:-1])/1
    uz_x,uz_y,uz_z = tf.math.add(u[2,1:,:-1,:-1], -u[2,:-1,:-1,:-1])/1,tf.math.add(u[2,:-1,1:,:-1], -u[2,:-1,:-1,:-1])/1,tf.math.add(u[2,:-1,:-1,1:], -u[2,:-1,:-1,:-1])/1
    exx = ux_x
    eyy = uy_y
    ezz = uz_z
    exy = 0.5*tf.math.add(ux_y,uy_x)
    exz = 0.5*tf.math.add(ux_z,uz_x)
    eyz = 0.5*tf.math.add(uy_z,uz_y)
    
    strain = tf.stack([exx,eyy,ezz,eyz,exz,exy])/step_size
#     strain = tf.pad(strain,[[0,0],[1,1],[1,1],[1,1]])
    strain = tf.pad(strain,[[0,0],[0,1],[0,1],[0,1]])

    return strain.numpy()

def plot_strain(strain,min_max=False,strn=True):
    fig,axs = plt.subplots(nrows=1,ncols=6,figsize=(30,4))
    c = np.array(strain.shape)[1]//2
    if strn:
        labels =[r'$\varepsilon_{xx}$',r'$\varepsilon_{yy}$',r'$\varepsilon_{zz}$',r'$\varepsilon_{yz}$',r'$\varepsilon_{xz}$',r'$\varepsilon_{xy}$'] 
    else:  
        labels = [r'$S_{xx}$',r'$S_{yy}$',r'$S_{zz}$',r'$S_{yz}$',r'$S_{xz}$',r'$S_{xy}$']
    for i,ax in enumerate(axs.ravel()):
        if min_max != False:
            minn,maxx = min_max
        else:
            minn,maxx = strain[:,c-5:c+5,c-5:c+5,c-5:c+5].min(),strain[:,c-5:c+5,c-5:c+5,c-5:c+5].max()
        A = ax.imshow(strain[i,:,:,c],vmin=minn,vmax=maxx,cmap='coolwarm')
        ax.set_title(labels[i])
        fig.colorbar(A,ax=ax)
    plt.show()
    return
def plot_u(u,minmax = None,axis=2):
    fig,axs = plt.subplots(nrows=1,ncols=3,figsize= (15,4))
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
def calc_u_vol(scans,gs,specfile,a,shape):
    phases = []
    x,y,z = np.meshgrid(np.arange(shape[0]),np.arange(shape[1]),np.arange(shape[1]),indexing='ij')
    for g,scan in zip(gs,scans):
        vals = parse_spec(specfile,int(scan[:3]))

        delta = vals[0]
        gamma = vals[1]
        arm = vals[7]
        energy = vals[-1]
        
        
        del_delt = np.degrees(np.arctan2(128*np.sqrt(2)*55e-6,arm))
        lam = 1239.84/(energy*1000)
        
        d_0 = a/np.linalg.norm(np.array(g))
        

        two_theta = np.arccos(np.cos(delta/180*np.pi)*np.cos(gamma/180*np.pi))+del_delt/180*np.pi
        d_1 = lam/(2*np.sin(two_theta/2))
        del_d = d_1 - d_0
        
        gs = np.array(gs)
        
        phase_vol = del_d*(g@np.stack([x,y,z]).reshape(3,-1)) 
        phases.append(phase_vol)
        
    gs = np.array(gs)    
    A = gs.T@gs
    b = gs.T@np.stack(phases)
    u_vol = np.linalg.solve(A,b).reshape((3,shape[0],shape[1],shape[2]))
        
        
    
    
    return u_vol
def calc_stress(strain,stiff,o_mats):
    o_mats = np.repeat(o_mats[np.newaxis,:,:],1,axis=0)
    shp = strain.shape
    strain = np.array([[strain[0],strain[5],strain[4]],
                       [strain[5],strain[1],strain[3]],
                       [strain[4],strain[3],strain[2]]])
    strain = np.moveaxis(strain.reshape((3,3,-1)),-1,0)

    
    strain = np.transpose(o_mats,axes=(0,2,1))@strain@o_mats
    
    
    strain = np.stack([strain[:,0,0],strain[:,1,1],strain[:,2,2],strain[:,1,2],strain[:,0,2],strain[:,0,1]])
    
    
    
    strain[3:,:] *=2    

    stress = stiff@strain#np.einsum('ilmn,ij->jlmn',strain,stiffness)

    stress = np.array([[stress[0],stress[5],stress[4]],
                   [stress[5],stress[1],stress[3]],
                   [stress[4],stress[3],stress[2]]])
    stress = np.moveaxis(stress.reshape((3,3,-1)),-1,0)
    
    stress = o_mats@stress@np.transpose(o_mats,axes=(0,2,1))

    stress = np.array([stress[:,0,0],stress[:,1,1],stress[:,2,2],stress[:,1,2],stress[:,0,2],stress[:,0,1]]).reshape(shp)
    
    return stress