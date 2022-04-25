'# -*- coding: utf-8 -*-'
"""
Created on Tue Aug 20 10:47:01 2019

@author: mattw
"""

from scipy.ndimage.interpolation import rotate
import numpy as np
from numpy.fft import fftn, fftshift
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# import Rotate_Peaks as RPs
from miscellaneous import translate
    
class f_model:
    
    def __init__(self,support,u,a,plot=True):

        self.grain = support
        
        self.a = a
        ux,uy,uz = u
        self.dim = support.shape
        # self.xx,self.yy,self.zz = np.meshgrid(ux,uy,uz)
        self.u_x = ux#*np.absolute(self.grain)#*(0.1*self.xx +
        self.u_y = uy#*np.absolute(self.grain)#*self.yy
        self.u_z = uz#*np.absolute(self.grain)#*(self.xx+0.3*self.yy)
        
        if plot:
            fig,ax = plt.subplots(ncols=3,figsize=(15,5))

            A = ax[0].imshow(self.u_x[:,:,self.dim[2]//2])
            B = ax[1].imshow(self.u_y[:,:,self.dim[2]//2])
            C = ax[2].imshow(self.u_z[:,:,self.dim[2]//2])
            ax[0].set_title('$u_x$')
            ax[1].set_title('$u_y$')
            ax[2].set_title('$u_z$')
            fig.colorbar(A,ax=ax[0])
            fig.colorbar(B,ax=ax[1])
            fig.colorbar(C,ax=ax[2])
            plt.savefig('Ground_Truth.png')
            plt.show()
        return 
       
    
    def forward(self,pln,plot=True,center=False):
        
        p = pln[0]*self.u_x+pln[1]*self.u_y+pln[2]*self.u_z
        phase = self.grain*np.exp(2j*p*np.pi/self.a)
        phase[self.grain==0] = 0. + 0.0j
        
        new_array = fftshift(fftn(fftshift(phase)))
        det_pat = np.absolute(new_array)**2

        output = det_pat
        if center:
            max_where = np.array(np.where(output==output.max()))
            max_where = [m[0] for m in max_where]
    #         print(max_where)
            output = translate(output,max_where,[self.dim[0]//2,self.dim[1]//2,self.dim[2]//2])
        
        

        if plot:
            ph = np.angle(phase)
            fig,ax = plt.subplots(nrows=1,ncols=2,figsize= (10,5))

            D = ax[0].imshow(output[:,:,self.dim[2]//2],norm=LogNorm())
            E = ax[1].imshow(ph[:,:,self.dim[2]//2])

            ax[0].set_title('Bragg Peak')
            ax[1].set_title('Phase_%s'%str(pln))

            fig.colorbar(D,ax=ax[0])
            fig.colorbar(E,ax=ax[1])
            plt.show()
        
        return output,phase
        
    

