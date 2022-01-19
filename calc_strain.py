# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:16:27 2020

@author: mattw
"""
import numpy as np
import matplotlib.pyplot as plt
def calc_strain(u,sup,del_x,plot=True,plot_range=[-0.003,0.003]):
    dif_x = np.where(np.diff(sup,axis=0,append=0)!=0,1,0)
    dif_y = np.where(np.diff(sup,axis=1,append=0)!=0,1,0)
    dif_z = np.where(np.diff(sup,axis=2,append=0)!=0,1,0)
    
    border = dif_x+dif_y+dif_z

    border= np.where(border==0,1,0)
#     border[np.where(sup==0)]=0
    
    shape = sup.shape
    ux,uy,uz = u
    ux_x,ux_y,ux_z = np.diff(ux,axis=0,append=0),np.diff(ux,axis=1,append=0),np.diff(ux,axis=2,append=0)
    uy_x,uy_y,uy_z = np.diff(uy,axis=0,append=0),np.diff(uy,axis=1,append=0),np.diff(uy,axis=2,append=0)
    uz_x,uz_y,uz_z = np.diff(uz,axis=0,append=0),np.diff(uz,axis=1,append=0),np.diff(uz,axis=2,append=0)
    
    exx = ux_x*border
    eyy = uy_y*border
    ezz = uz_z*border
    exy = 0.5*(ux_y + uy_x)*border
    exz = 0.5*(ux_z + uz_x)*border
    eyz = 0.5*(uy_z + uz_y)*border
    
    strain = np.array([[exx,exy,exz],[exy,eyy,eyz],[exz,eyz,ezz]])/del_x
    if plot:
        fig,ax = plt.subplots(nrows=2,ncols=3,figsize=(18,10))
        ind = [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]
        labels = ['$\epsilon_{xx}$','$\epsilon_{yy}$','$\epsilon_{zz}$','$\epsilon_{xy}$','$\epsilon_{xz}$','$\epsilon_{yz}$']
        c=0
        s2 = [strain[ind[0]][:,:,shape[2]//2],strain[ind[1]][:,:,shape[2]//2],strain[ind[2]][shape[2]//2,:,:],
             strain[ind[3]][:,:,shape[2]//2],strain[ind[4]][:,:,shape[2]//2],strain[ind[5]][:,:,shape[2]//2]]
        for i in range(2):
            for j in range(3):
                A = ax[i,j].imshow(s2[c],vmin=plot_range[0],vmax=plot_range[1],cmap='bwr')
                ax[i,j].set_title(labels[c])
                t = fig.colorbar(A,ax=ax[i,j])

                c+=1
        
        plt.show()
    return strain