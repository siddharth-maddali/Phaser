from shrinkwrap import Shrinkwrap
import sys

import Phaser as ph
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class cpr:
    
    """
    Class similar to phaser
    
    inputs:
        data -- stack of datasets (numpy array, dtype=float,dimensions=(num_datasetsxnxnxn))
        qs -- q vectors for reflections corresponding to the datasets in data (numpy array, dtype=float,dimensions=(num_datasetsx3)) 
        sup -- guess for object support (numpy array, dtype=float,dimensions=(nxnxn)) 
        amp -- guess for amplitude (numpy array, dtype=float,dimensions=(nxnxn)). Must be specified if random_start = False.
        u -- guess for displacement (numpy array, dtype=float,dimensions=(3xnxnxn)). Must be specified if random_start = False.
        a -- lattice parameter (dtype=float units: nm)
        random_start -- option to start with random phase on all constituents. If true, u is ignored (boolean)
        bad_pix -- hacky fix to phase wraps. If a certain number of pixels are close to pi or -pi, we add 2*pi to those pixels (dtype int)
        pcc -- option to turn on partial coherence correction (dtype=boolean)
        gpu -- option to use gpu or not (dtype=boolean)
        
    functions:
        run_recipes -- runs multi_phaser recipes (see tutorial.ipynb for details) --> example: [['ER:20',[1.0,0.1]],['HIO:20',[0]]] 
    attributes:
        extract_vals -- returns dictionary containing u,amp,sup
        extract_error -- returns two error lists over all iterations, one for chi^2 and one for least squares loss
    
    """
    def __init__(self,data,qs,sup,a,amp=False,u=False,random_start=True,bad_pix=200,pcc=False,gpu=True):
        if random_start:
            u = np.repeat(sup[np.newaxis,:,:,:],3,axis=0)
            amp = sup
            
        self.bad_pix = bad_pix
        self.u = tf.Variable(u,dtype=tf.float32)
        self.sup = tf.Variable(sup,dtype=tf.float32)
        self.amp = tf.Variable(amp,dtype=tf.float32)
        self.a = tf.constant(a,dtype=tf.float32)
        self.Qs = tf.stack([tf.constant(q,dtype=tf.float32) for q in qs])*tf.constant(2.0*np.pi,dtype=tf.float32)/self.a
        self.dim = data[0].shape
        self.span = [s//2 for s in self.dim]
        self.paddings = 0
        phases = tf.matmul(self.Qs,tf.reshape(self.u,(3,-1))) 
        phases = tf.reshape(phases,(self.Qs.shape[0],self.dim[0],self.dim[1],self.dim[2])).numpy()
        imgs = [self.amp.numpy()*np.exp(phase*1j) for phase in phases] 

        self.recons = [ph.Phaser(
            modulus = np.sqrt( data[i] ),
            support = sup,
             img_guess = imgs[i],
            random_start = random_start,pcc=pcc,gpu=gpu).gpusolver for i in range(len(data))]

        
        self._error = []
        self.L = []
        self._poisson_log = []

              
        
    def extract_error( self ):
        self._error = tf.reduce_sum(tf.stack([r._error[1:] for r in self.recons]),axis=0)
#         self._poisson_log= tf.reduce_sum(tf.stack([r._poisson_log[1:] for r in self.recons]),axis=0)
        
        return list(self._error.numpy()), list(self.L)
    def extract_vals(self):
        self.u = tf.stack([tf.pad(s,self.paddings,'constant') for s in self.u])
        self.sup = tf.pad(self.sup,self.paddings,'constant')
        self.amp = tf.pad(self.amp,self.paddings,'constant')
        
        return {'u':self.u.numpy()*self.sup.numpy()[np.newaxis,:,:,:],'amp':self.amp.numpy(),'sup':self.sup.numpy()}

    def UpdateSupport( self ):

        for i,r in enumerate(self.recons):
            r.UpdateSupport(tf.pad(self.sup,self.paddings,'constant'))

    def center_phase(self):
        for r in self.recons:
            supp = tf.cast(r._support,dtype=tf.float32)
            phase = tf.cast(tf.math.angle(r._cImage),dtype=tf.float32)
            amp = tf.abs(r._cImage)
            phase *= supp


            if tf.math.count_nonzero(phase[phase>3.1]) >self.bad_pix:

                phase= tf.where(phase<0.,phase+2*tf.constant(np.pi),phase)
            phase *= supp
            phase -= tf.math.reduce_sum(phase)/tf.math.reduce_sum(supp)
            phase = tf.cast(phase,dtype=tf.complex64)
            amp = tf.cast(amp,dtype=tf.complex64)
            r._cImage = tf.Variable(amp*tf.math.exp(phase*1j),dtype=tf.complex64)

        return




    

    
    def phase_to_u(self,sw):
        self.energies = tf.stack([tf.norm(tf.abs(r._cImage)) for r in self.recons])
        
        self.amp = tf.math.reduce_mean( tf.stack([tf.abs(r._cImage)/tf.reduce_sum(tf.abs(r._cImage)) for r in self.recons]),axis=0)
        phs = tf.stack([tf.math.angle(r._cImage) for r in self.recons])
        phs = tf.transpose(phs,[1,2,3,0])
        shp = phs.shape
        phs = tf.reshape(phs,(shp[0],shp[1],shp[2],shp[3],1))
        
        small_sup = Shrinkwrap(self.amp,1.0,0.2)
        
        
        if sw[0] != 0:
            
            self.sup = tf.Variable(Shrinkwrap(self.amp,sw[0],sw[1])) 
            inds = tf.where(self.sup==1)
           
            pad = 2
            max_span = max([max([shp[i]//2-tf.math.reduce_min(inds[:,i]),tf.math.reduce_max(inds[:,i])-shp[i]//2]) for i in range(3)])
            self.span = [max_span + pad for i in range(3)]
            


        

        phs = phs[shp[0]//2-self.span[0]:shp[0]//2+self.span[0],shp[1]//2-self.span[1]:shp[1]//2+self.span[1],shp[2]//2-self.span[2]:shp[2]//2+self.span[2],:,:]
        
        self.amp = self.amp[shp[0]//2-self.span[0]:shp[0]//2+self.span[0],shp[1]//2-self.span[1]:shp[1]//2+self.span[1],shp[2]//2-self.span[2]:shp[2]//2+self.span[2]]
        self.sup = self.sup[self.sup.shape[0]//2-self.span[0]:self.sup.shape[0]//2+self.span[0],self.sup.shape[1]//2-self.span[1]:self.sup.shape[1]//2+self.span[1],self.sup.shape[2]//2-self.span[2]:self.sup.shape[2]//2+self.span[2]]
        small_sup = small_sup[shp[0]//2-self.span[0]:shp[0]//2+self.span[0],shp[1]//2-self.span[1]:shp[1]//2+self.span[1],shp[2]//2-self.span[2]:shp[2]//2+self.span[2]]


        
        
        self.amp *= self.sup
        self.amp = self.amp/tf.math.reduce_sum(self.amp)     
        
        self.shp1=phs.shape[:4]
        qs = tf.ones((self.shp1[0],self.shp1[1],self.shp1[2],self.shp1[3],3))*self.Qs
        self.u = tf.linalg.lstsq( qs, phs, l2_regularizer=0.0, fast=True, name=None)
        norm_L = tf.reduce_sum((tf.matmul(qs,self.u)-phs)**2,axis=3)
        
        
        
        norm_L = norm_L[:,:,:,0]*self.amp*small_sup
        
        
        L = (tf.reduce_sum(norm_L)/tf.reduce_sum(small_sup)).numpy()
        self.L.append(L)
        self.u = tf.reshape(self.u,(self.shp1[0],self.shp1[1],self.shp1[2],3))
        self.u = tf.transpose(self.u,[3,0,1,2])
        paddings = [(self.dim[i]-self.shp1[i])//2 for i in range(3)]
       
        self.paddings = [[p,p] for p in paddings]
                         
        return
   
    def u_to_phase(self):        
        u = tf.reshape(self.u,(3,-1))
        
        phase = tf.matmul(self.Qs,u) 
        
        phase = tf.reshape(phase,(self.Qs.shape[0],self.shp1[0],self.shp1[1],self.shp1[2]))
        
        phase = tf.pad(phase,[[0,0],self.paddings[0],self.paddings[1],self.paddings[2]],'constant')
        
        energies = tf.cast(self.energies,dtype=tf.complex64)
        amp = tf.pad(self.amp,self.paddings,'constant')
        amp = tf.cast(amp/tf.norm(amp),dtype=tf.complex64)
        
        phase = tf.cast(phase,dtype=tf.complex64)

        for i,r in enumerate(self.recons):
            r._cImage = tf.Variable(amp*energies[i]*tf.math.exp(phase[i]*1j),dtype=tf.complex64)
        return   
        
    def run_recipes(self,recipes):
        for i,recipe in enumerate(recipes):
            self.iteration = i
            
            for r in self.recons: 
                r.runRecipe(recipe[0])   
                if i%10 ==0:
                    r.Retrieve()
            if i<5:
                self.center_phase()
            self.phase_to_u(recipe[1])
            self.UpdateSupport()
            self.u_to_phase()

        
        return
    

    