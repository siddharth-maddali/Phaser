# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:25:33 2020

@author: mattw
"""




from scipy.ndimage.interpolation import rotate
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator as RGI
import matplotlib.pyplot as plt


        
        
def interpolate(data,Bases,step_size,plot=False):
    B_inv = np.linalg.inv(Bases)
    shape = data.shape
    
    x = np.arange(-shape[0]//2,shape[0]//2)     #define old grid
    y = np.arange(-shape[1]//2,shape[1]//2)
    z = np.arange(-shape[2]//2,shape[2]//2)
    xx,yy,zz = np.meshgrid(x,y,z,indexing='ij')
    old_grid = np.vstack([xx.ravel(),yy.ravel(),zz.ravel()])  #flatten grid into a 3xN array

    new_points = Bases@old_grid  #find coordinates of new grid

    # FIND BOUNDS OF NEW GRID
    min_max = np.zeros((3,2))
    for j in range(3):
        
        min_max[j,:] = [new_points[j,:].min(),new_points[j,:].max()]  #Find smallest and largest values in each dimension

                    #calculate edge length of unit cell
    
    a = np.arange(min_max[0,0],min_max[0,1],step_size)     #define new grid from the smallest to largest value
    b = np.arange(min_max[1,0],min_max[1,1],step_size)     # ensure that the step size is that of the unit cell edge length
    c = np.arange(min_max[2,0],min_max[2,1],step_size)
    aa,bb,cc = np.meshgrid(a,b,c,indexing='ij')

    new_grid = np.vstack([aa.ravel(),bb.ravel(),cc.ravel()])  #flatten new grid
    interp_points = B_inv@new_grid                            #transform new grid points to old grid
    

              #values at each point in data

    
    start = time.time()
    interp = RGI((x,y,z),data,fill_value=0,bounds_error=False)    #feed interpolator node values from data
    new_vals = interp(interp_points.T)     #interpolate values at new grid points
    
    data2 = new_vals.reshape((a.shape[0],b.shape[0],c.shape[0]))     #reshape data array back to the original dimensions

    end = time.time()
    print('time to calculate:',end-start)
    data[np.isnan(data)]=0
    data2[np.isnan(data2)]=0
    if plot:
        fig,ax = plt.subplots(nrows = 2,ncols=3,figsize=(15,10))
        ax[0,0].imshow(data[:,:,shape[2]//2],norm=LogNorm())
        ax[0,1].imshow(data[:,shape[1]//2,:],norm=LogNorm())
        ax[0,2].imshow(data[shape[0]//2,:,:],norm=LogNorm())
        ax[1,0].imshow(data2[:,:,data2.shape[2]//2],norm=LogNorm())
        ax[1,1].imshow(data2[:,data2.shape[1]//2,:],norm=LogNorm())
        ax[1,2].imshow(data2[data2.shape[0]//2,:,:],norm=LogNorm())    
        plt.show()
        

    return data2





def rotate(data,Bases,step_size,plot=False):
    B_inv = np.linalg.inv(Bases)
    shape = data.shape
    
    x = np.arange(-shape[0]//2,shape[0]//2)     #define old grid
    y = np.arange(-shape[1]//2,shape[1]//2)
    z = np.arange(-shape[2]//2,shape[2]//2)
    xx,yy,zz = np.meshgrid(x,y,z,indexing='ij')
    old_grid = np.vstack([xx.ravel(),yy.ravel(),zz.ravel()])  #flatten grid into a 3xN array

    new_points = Bases@old_grid  #find coordinates of new grid
    
    #FIND BOUNDS OF NEW GRID
    min_max = np.zeros((3,2))
    for j in range(3):
        
        min_max[j,:] = [new_points[j,:].min(),new_points[j,:].max()]  #Find smallest and largest values in each dimension

                    #calculate edge length of unit cell
    
    a = np.arange(min_max[0,0],min_max[0,1],step_size)     #define new grid from the smallest to largest value
    b = np.arange(min_max[1,0],min_max[1,1],step_size)     # ensure that the step size is that of the unit cell edge length
    c = np.arange(min_max[2,0],min_max[2,1],step_size)
    aa,bb,cc = np.meshgrid(a,b,c,indexing='ij')

    new_grid = np.vstack([aa.ravel(),bb.ravel(),cc.ravel()])  #flatten new grid
    interp_points = B_inv@new_grid                            #transform new grid points to old grid
    

              #values at each point in data

    
    start = time.time()
    interp = RGI((x,y,z),data,fill_value=0,bounds_error=False)    #feed interpolator node values from data
    new_vals = interp(interp_points.T)     #interpolate values at new grid points
    
    data2 = new_vals.reshape((a.shape[0],b.shape[0],c.shape[0]))     #reshape data array back to the original dimensions

    end = time.time()
    print('time to calculate:',end-start)
    data[np.isnan(data)]=0
    data2[np.isnan(data2)]=0
    if plot:
        fig,ax = plt.subplots(nrows = 2,ncols=3,figsize=(15,10))
        ax[0,0].imshow(data[:,:,shape[2]//2])
        ax[0,1].imshow(data[:,shape[1]//2,:])
        ax[0,2].imshow(data[shape[0]//2,:,:])
        ax[1,0].imshow(data2[:,:,data2.shape[2]//2])
        ax[1,1].imshow(data2[:,data2.shape[1]//2,:])
        ax[1,2].imshow(data2[data2.shape[0]//2,:,:])    
        plt.show()
        

    return data2