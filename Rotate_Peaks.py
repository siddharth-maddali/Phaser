# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:25:33 2020

@author: mattw
"""




from scipy.ndimage.interpolation import rotate
import numpy as np
from scipy.spatial.transform import Rotation as R

    

        
        



def rotate_peak(original,theta,chi,phi,direction = 'Forward',space='Direct'):
    
    """Transforms object from Lab Frame to Sample Frame ('Forward')
        or from Sample Frame to Lab Frame ('Backward')

        Uses extrinsic eulerian rotation to take advantage of the fast scipy.ndimage
        rotation/interpolation package"""

    dim = original.shape
    r = R.from_euler('YZY',[phi,-chi,theta])     #create Rotation object from intrinsic euler angles phi,chi,theta
    print("Intrinsic Euler Angles:",[phi,chi,theta])

    ex_euler = r.as_euler('yzy',degrees=True)    #convert to extrinsic euler angles
    print("Extrinsic Euler Angles:",ex_euler)
    
    axs = [(0,2),(0,1),(0,2)]      #define axes of "image" to be rotated. (0,2) means rotate about Y axis, (0,1) would be Z
    
    if direction == 'Backward':       #Sample to Lab. Take the negatives of phi,chi,theta and reverse the order
        ex_euler= -np.flip(ex_euler)
        axs = np.flip(axs)
    new_object = original.copy()
    low_value = np.sort(np.unique(original.ravel()))[1]
    

    for i in range(3):
        
        new_object = rotate(new_object,ex_euler[i],axes=axs[i],order=3)   #rotate object
              
        
        new_object = new_object[new_object.shape[0]//2-dim[0]//2:new_object.shape[0]//2+dim[0]//2,
                        new_object.shape[1]//2-dim[1]//2:new_object.shape[1]//2+dim[1]//2,
                        new_object.shape[2]//2-dim[2]//2:new_object.shape[2]//2+dim[2]//2]   #crop image to original dimensions
        
        if space == 'Reciprocal':
            new_object[new_object<low_value] = low_value


    return new_object


