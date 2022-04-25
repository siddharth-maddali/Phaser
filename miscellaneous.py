import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from copy import copy
from spec import parse_spec
from shrinkwrap2 import Shrinkwrap
from scipy import ndimage

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm


def rescale_noise(pks,max_val):
    peaks = []
    for m in range(len(pks)):
        pk = pks[m]
        pk = pk/pk.max()
        pk *= max_val
        
        pk = np.random.poisson(pk)
        # pk = np.round(pk,0)


        pk[pk<0] = 0
        peaks.append(pk)
    return peaks


def center_u(u,support):
    support = support == 1
    for i in range(3):
        t = copy(u[i])
        t[support] -= t[support].mean()
        u[i] = t
    return u

def calc_strain_cpu(u,sup,step):
    dif_x = np.absolute(np.diff(sup,axis=0,append=0))
    dif_y = np.absolute(np.diff(sup,axis=1,append=0))
    dif_z = np.absolute(np.diff(sup,axis=2,append=0))
    
    border = dif_x+dif_y+dif_z
    
    border= np.where(border==0.,1.,0.)
#     border[np.where(sup==0)]=0
    
    shape = sup.shape
    ux,uy,uz = u
    ux_x,ux_y,ux_z = np.diff(ux,axis=0,append=0),np.diff(ux,axis=1,append=0),np.diff(ux,axis=2,append=0)
    uy_x,uy_y,uy_z = np.diff(uy,axis=0,append=0),np.diff(uy,axis=1,append=0),np.diff(uy,axis=2,append=0)
    uz_x,uz_y,uz_z = np.diff(uz,axis=0,append=0),np.diff(uz,axis=1,append=0),np.diff(uz,axis=2,append=0)
    
    exx = ux_x
    eyy = uy_y
    ezz = uz_z
    exy = 0.5*(ux_y + uy_x)
    exz = 0.5*(ux_z + uz_x)
    eyz = 0.5*(uy_z + uz_y)
    
    strain = np.array([exx,eyy,ezz,eyz,exz,exy])/step*border
    return strain
def calc_strain_gpu(u,sup,step_size):
    sup = tf.constant(sup,dtype=tf.float32)
    u = tf.constant(u,dtype=tf.float32)
    bx = tf.abs(tf.math.add(sup[1:,:-1,:-1], -sup[:-1,:-1,:-1]))
    by = tf.abs(tf.math.add(sup[:-1,1:,:-1], -sup[:-1,:-1,:-1]))
    bz = tf.abs(tf.math.add(sup[:-1,:-1,1:], -sup[:-1,:-1,:-1]))
    border = bx+by+bz
    border = tf.where(border != 0.,0.,1.)
    
    
    ux_x,ux_y,ux_z = tf.math.add(u[0,1:,:-1,:-1], -u[0,:-1,:-1,:-1]),tf.math.add(u[0,:-1,1:,:-1], -u[0,:-1,:-1,:-1]),tf.math.add(u[0,:-1,:-1,1:], -u[0,:-1,:-1,:-1])
    uy_x,uy_y,uy_z = tf.math.add(u[1,1:,:-1,:-1], -u[1,:-1,:-1,:-1]),tf.math.add(u[1,:-1,1:,:-1], -u[1,:-1,:-1,:-1]),tf.math.add(u[1,:-1,:-1,1:], -u[1,:-1,:-1,:-1])
    uz_x,uz_y,uz_z = tf.math.add(u[2,1:,:-1,:-1], -u[2,:-1,:-1,:-1]),tf.math.add(u[2,:-1,1:,:-1], -u[2,:-1,:-1,:-1]),tf.math.add(u[2,:-1,:-1,1:], -u[2,:-1,:-1,:-1])
    exx = ux_x
    eyy = uy_y
    ezz = uz_z
    exy = 0.5*tf.math.add(ux_y,uy_x)
    exz = 0.5*tf.math.add(ux_z,uz_x)
    eyz = 0.5*tf.math.add(uy_z,uz_y)
    
    strain = tf.math.multiply(border,tf.stack([exx,eyy,ezz,eyz,exz,exy]))/step_size
#     strain = tf.pad(strain,[[0,0],[1,1],[1,1],[1,1]])
    strain = tf.pad(strain,[[0,0],[0,1],[0,1],[0,1]])

    return strain.numpy()
def unpack_obj(obj):
    amp = np.absolute(obj)[0]
    sup = np.where(amp>amp.max()*0.01,1,0)
#     sup = Shrinkwrap(amp,1.0,0.1).numpy()
    u = np.angle(obj)*sup
    return u,amp,sup

def calc_rot(u,sup,step_size):
    sup = tf.constant(sup,dtype=tf.float32)
    u = tf.constant(u,dtype=tf.float32)
    bx = tf.abs(tf.math.add(sup[1:,:-1,:-1], -sup[:-1,:-1,:-1]))
    by = tf.abs(tf.math.add(sup[:-1,1:,:-1], -sup[:-1,:-1,:-1]))
    bz = tf.abs(tf.math.add(sup[:-1,:-1,1:], -sup[:-1,:-1,:-1]))
    border = bx+by+bz
    border = tf.where(border != 0.,0.,1.)
    ux_x,ux_y,ux_z = tf.math.add(u[0,1:,:-1,:-1], -u[0,:-1,:-1,:-1])/1,tf.math.add(u[0,:-1,1:,:-1], -u[0,:-1,:-1,:-1])/1,tf.math.add(u[0,:-1,:-1,1:], -u[0,:-1,:-1,:-1])/1
    uy_x,uy_y,uy_z = tf.math.add(u[1,1:,:-1,:-1], -u[1,:-1,:-1,:-1])/1,tf.math.add(u[1,:-1,1:,:-1], -u[1,:-1,:-1,:-1])/1,tf.math.add(u[1,:-1,:-1,1:], -u[1,:-1,:-1,:-1])/1
    uz_x,uz_y,uz_z = tf.math.add(u[2,1:,:-1,:-1], -u[2,:-1,:-1,:-1])/1,tf.math.add(u[2,:-1,1:,:-1], -u[2,:-1,:-1,:-1])/1,tf.math.add(u[2,:-1,:-1,1:], -u[2,:-1,:-1,:-1])/1

    omz = 0.5*tf.math.add(ux_y,-uy_x)
    omy = 0.5*tf.math.add(ux_z,-uz_x)
    omx = 0.5*tf.math.add(uy_z,-uz_y)
    
    strain = tf.math.multiply(border,tf.stack([omx,omy,omz]))/step_size
#     strain = tf.pad(strain,[[0,0],[1,1],[1,1],[1,1]])
    strain = tf.pad(strain,[[0,0],[0,1],[0,1],[0,1]])

    return strain.numpy()

def stack_strain(strain):
    exx,eyy,ezz,eyz,exz,exy = strain
    strain = np.array([[exx,exy,exz],
                       [exy,eyy,eyz],
                       [exz,eyz,ezz]])
    return strain
def stack_rot(rot):
    omx,omy,omz = rot
    rot_mat = np.array([[0,omz,omy],
                       [-omz,0,omx],
                       [-omy,-omx,0]])
    return rot_mat


def plot_strain(strain,minmax=False,strn=True,cmap='bwr',shp=(1,6),axis=2):
    fig,axs = plt.subplots(nrows=shp[0],ncols=shp[1],figsize=(6*shp[1],4*shp[0]))
    c = np.array(strain.shape)[1]//2
    if strn:
        labels =[r'$\varepsilon_{xx}$',r'$\varepsilon_{yy}$',r'$\varepsilon_{zz}$',r'$\varepsilon_{yz}$',r'$\varepsilon_{xz}$',r'$\varepsilon_{xy}$'] 
    else:  
        labels = [r'$S_{xx}$',r'$S_{yy}$',r'$S_{zz}$',r'$S_{yz}$',r'$S_{xz}$',r'$S_{xy}$']
    for i,ax in enumerate(axs.ravel()):
        if minmax != False:
            minn,maxx = minmax
        else:
            minn,maxx = strain[:,c-5:c+5,c-5:c+5,c-5:c+5].min(),strain[:,c-5:c+5,c-5:c+5,c-5:c+5].max()
            
        if axis == 0:
            
            A = ax.imshow(strain[i,c,:,:],vmin=minn,vmax=maxx,cmap=cmap)
        if axis == 1:
            A = ax.imshow(strain[i,:,c,:],vmin=minn,vmax=maxx,cmap=cmap)
        if axis == 2:
            A = ax.imshow(strain[i,:,:,c],vmin=minn,vmax=maxx,cmap=cmap)
        ax.set_title(labels[i],fontsize=20)
        fig.colorbar(A,ax=ax,format='%.0e')
    plt.show()
    return 
def plot_u(u,minmax = None,axis=2,cmap='viridis',scale = None,ax_ticks=True,show = True):
    
    fig,axs = plt.subplots(nrows=1,ncols=3,figsize= (15,4))
    fontprops = fm.FontProperties(size=16)
    c = np.array(u.shape)//2
    labels = ['$u_x$','$u_y$','$u_z$']
    
    
    for i,ax in enumerate(axs):
        s = u[i,c[1]-5:c[1]+5,c[2]-5:c[2]+5,c[3]-5:c[3]+5]   
        if minmax == None:
            if axis == 0:
                
                A = ax.imshow(u[i,c[1],:,:].T,cmap=cmap,origin='lower')
            if axis == 1:
                A = ax.imshow(u[i,:,c[2],:].T,cmap=cmap,origin='lower')
            if axis == 2:
                A = ax.imshow(u[i,:,:,c[3]].T,cmap=cmap,origin='lower')
        else:
            if axis == 0:
                A = ax.imshow(u[i,c[1],:,:].T,vmin=minmax[0],vmax=minmax[1],cmap=cmap,origin='lower')
            if axis == 1:
                A = ax.imshow(u[i,:,c[2],:].T,vmin=minmax[0],vmax=minmax[1],cmap=cmap,origin='lower')
            if axis == 2:
                A = ax.imshow(u[i,:,:,c[3]].T,vmin=minmax[0],vmax=minmax[1],cmap=cmap,origin='lower')
        if scale is not None:
            scalebar = AnchoredSizeBar(ax.transData,
                       10, '%s nm'%scale, 'lower left', 
                       pad=0.1,
                       color='black',
                       frameon=False,
                       size_vertical=1,
                       fontproperties=fontprops)
            ax.add_artist(scalebar) 
        if ax_ticks == False:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

        ax.set_title(labels[i],fontsize=16)
        cbar = fig.colorbar(A,ax=ax,shrink=0.9)
        cbar.set_label('nm', rotation=270)
    if show:
        plt.show()
    return fig
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



def transform_tensor(tensor,o_mat,reshape=True):
    o_mat = o_mat[np.newaxis,:,:]
    shp = tensor.shape
    tensor = np.array([[tensor[0],tensor[5],tensor[4]],
                   [tensor[5],tensor[1],tensor[3]],
                   [tensor[4],tensor[3],tensor[2]]])
    tensor = np.moveaxis(tensor.reshape((3,3,-1)),-1,0)
    o_mat@tensor@np.transpose(o_mat,axes=(0,2,1))
    tensor = np.array([tensor[:,0,0],tensor[:,1,1],tensor[:,2,2],tensor[:,1,2],tensor[:,0,2],tensor[:,0,1]])
    if reshape:
        tensor = tensor.reshape(shp)
    return tensor

def calc_stress(strain,stiff,o_mat):
    shp = strain.shape
    strain = transform_tensor(strain,o_mat.T,reshape=False)
    
    strain[3:,:] *=2    

    stress = stiff@strain
    
    stress = transform_tensor(stress,o_mat,reshape=False)

    return stress.reshape(shp)

def plot_rotation(u,minmax = None,axis=2):
    fig,axs = plt.subplots(nrows=1,ncols=3,figsize= (15,4))
    c = np.array(u.shape)//2
    labels = [r'$\omega_x$',r'$\omega_y$',r'$\omega_z$']
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

def block_mean(ar, fact):
    
    
    
    sx, sy, sz = ar.shape
    X, Y, Z = np.ogrid[0:sx, 0:sy, 0:sz]
    regions = sz//fact[2]*sy//fact[1] * (X//fact[0]) + sz//fact[2]*(Y//fact[1]) + Z//fact[2]
    res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx//fact[0], sy//fact[1],sz//fact[2])
    return res