from shrinkwrap import Shrinkwrap
import sys
import align_images as ai
import Phaser as ph
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from skimage.restoration import unwrap_phase
from miscellaneous import unpack_obj
from numpy.fft import fftn,fftshift
import yaml
class cpr:

    """
    Class similar to phaser
    
    inputs:
        data -- stack of datasets (numpy array, dtype=float,dimensions=(num_datasetsxnxnxn))
        qs -- q vectors for reflections corresponding to the datasets in data (numpy array, dtype=float,dimensions=(num_datasetsx3)) 
        a -- lattice parameter (dtype=float units: nm)
        obj -- guess for complex displacement object (numpy array, dtype=float,dimensions=(3xnxnxn)). Must be specified if random_start = False.
        
        random_start -- option to start with random phase on all constituents. If true, u is ignored (boolean)
        center -- option to phases of each constituent about zero (dtype=boolean)
        pcc -- option to turn on partial coherence correction (dtype=boolean)
        free_vox_mask -- numpy mask to determine which voxels are used for optimization (numpy array,dtype=boolean, dimensions = (nxnxn))
        gpu -- option to use gpu or not (dtype=boolean)
        unwrap -- option to unwrap phases (dtype=boolean)
        pcc_params -- partial coherence gaussian parameters, one for each constituent (numpy array,dytpe=float, dimensions = (num_datasetsx6))
        
    functions:
        run_recipes -- runs multi_phaser recipes (see tutorial.ipynb for details) --> example: [['ER:20',[1.0,0.1]],['HIO:20',[0]]] 
    attributes:
        extract_vals -- returns dictionary containing u,amp,sup
        extract_error -- returns two error lists over all iterations, one for chi^2 and one for least squares loss
    
    """
    def __init__(self,data,qs,a,obj=None,
                 random_start=True,
                 pcc=False,
                 gpu=True,
                 pcc_params=None,
                 unwrap=False,
                 center=False):
        
        

            
        
        if obj is not None:
            u,amp,sup = unpack_obj(obj)
        if random_start:
            sup = np.absolute(fftshift(fftn(data[0])))
            sup = np.where(sup>0.001*sup.max(),1,0)
            
            u = np.repeat(sup[np.newaxis,:,:,:],3,axis=0)
            amp = sup
        self.unwrap = unwrap
        self.center = center
        
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
            support = None if random_start else sup,
             img_guess = None if random_start else imgs[i],
            random_start = random_start,pcc=pcc,gpu=gpu).gpusolver for i in range(len(data))]
        if pcc_params != None:
            for recon,param in zip(self.recons,pcc_params):
                recon._pccSolver._vars = param
                
                
        self._error = []
        self.L = []
        self._poisson_log = []


    def extract_params(self):
        return [recon._pccSolver._vars for recon in self.recons]
        
    def extract_error( self ):
        self._error = tf.reduce_mean(tf.stack([r._error[1:] for r in self.recons]),axis=0)
        
#         self._poisson_log= tf.reduce_sum(tf.stack([r._poisson_log[1:] for r in self.recons]),axis=0)

        return list(self._error.numpy()), list(self.L)
    def extract_obj(self):
        self.u = tf.stack([tf.pad(s,self.paddings,'constant') for s in self.u]).numpy()
        center_u = self.u[:,self.dim[0]//2,self.dim[1]//2,self.dim[2]//2]
        self.u -= center_u[:,np.newaxis,np.newaxis,np.newaxis]
        self.sup = tf.pad(self.sup,self.paddings,'constant').numpy()
        self.amp = tf.pad(self.amp,self.paddings,'constant').numpy()
#         {'u':self.u.numpy()*self.sup.numpy()[np.newaxis,:,:,:],'amp':self.amp.numpy(),'sup':self.sup.numpy()}
        return  self.amp*self.sup*np.exp(self.u*1j)

    def UpdateSupport( self ):

        for i,r in enumerate(self.recons):
            r.UpdateSupport(tf.pad(self.sup,self.paddings,'constant'))

    def center_phase(self):
        for r in self.recons:
            supp = tf.cast(r._support,dtype=tf.float32)
            phase = tf.cast(tf.math.angle(r._cImage),dtype=tf.float32)
            amp = tf.abs(r._cImage)
            phase *= supp


            phase -= tf.math.reduce_sum(phase)/tf.math.reduce_sum(supp)
            phase = tf.cast(phase,dtype=tf.complex64)
            amp = tf.cast(amp,dtype=tf.complex64)
            r._cImage = tf.Variable(amp*tf.math.exp(phase*1j),dtype=tf.complex64)

        return







    def phase_to_u(self,sw,plot_amp):
        self.energies = tf.stack([tf.reduce_sum(tf.abs(r._cImage)) for r in self.recons])
        self.amp =  tf.stack([tf.abs(r._cImage)/tf.reduce_sum(tf.abs(r._cImage)) for r in self.recons])
        
        if plot_amp:
            fig,axs = plt.subplots(ncols=self.amp.shape[0],figsize = (15,4))
            for ax,amp in zip(axs,self.amp):
                ax.imshow(tf.transpose(amp[20:-20,20:-20,self.amp.shape[2]//2]))
            plt.show()

        self.amp = tf.math.reduce_mean(self.amp,axis=0)

        phs = [tf.math.angle(r._cImage)*tf.cast(r._support,dtype=tf.float32) for r in self.recons]
        small_sup = Shrinkwrap(self.amp,1.0,0.2)

        #option for phase unwrapping
        if self.unwrap:

            
            ss = np.array(self.sup.shape)//2
            for i in range(len(self.recons)):

                m = np.array(phs[i].shape)//2-ss
                p = phs[i][self.dim[0]//2-ss[0]:self.dim[0]//2+ss[0],self.dim[1]//2-ss[1]:self.dim[1]//2+ss[1],self.dim[2]//2-ss[2]:self.dim[2]//2+ss[2]]
                ph = tf.Variable(unwrap_phase(p.numpy(),seed=0),dtype=tf.float32)*self.sup

                phs[i] = tf.pad(ph,[[m[0],m[0]],[m[1],m[1]],[m[2],m[2]]],'constant')
                phs[i] -= tf.reduce_mean(phs[i][ss[0]-10:ss[0]+10,ss[1]-10:ss[1]+10,ss[2]-10:ss[2]+10])

        phs = tf.stack(phs)
        
        
        phs = tf.transpose(phs,[1,2,3,0])
        shp = phs.shape
        phs = tf.reshape(phs,(shp[0],shp[1],shp[2],shp[3],1))

        


        if sw[0] != 0:

            self.sup = tf.Variable(Shrinkwrap(self.amp,sw[0],sw[1])) 
            inds = tf.where(self.sup==1)

            pad = 1
            max_span = max([max([shp[i]//2-tf.math.reduce_min(inds[:,i]),tf.math.reduce_max(inds[:,i])-shp[i]//2]) for i in range(3)])
            self.span = [max_span + pad for i in range(3)]





        phs = phs[shp[0]//2-self.span[0]:shp[0]//2+self.span[0],shp[1]//2-self.span[1]:shp[1]//2+self.span[1],shp[2]//2-self.span[2]:shp[2]//2+self.span[2],:,:]

        self.amp = self.amp[shp[0]//2-self.span[0]:shp[0]//2+self.span[0],shp[1]//2-self.span[1]:shp[1]//2+self.span[1],shp[2]//2-self.span[2]:shp[2]//2+self.span[2]]
        self.sup = self.sup[self.sup.shape[0]//2-self.span[0]:self.sup.shape[0]//2+self.span[0],self.sup.shape[1]//2-self.span[1]:self.sup.shape[1]//2+self.span[1],self.sup.shape[2]//2-self.span[2]:self.sup.shape[2]//2+self.span[2]]
        small_sup = small_sup[shp[0]//2-self.span[0]:shp[0]//2+self.span[0],shp[1]//2-self.span[1]:shp[1]//2+self.span[1],shp[2]//2-self.span[2]:shp[2]//2+self.span[2]]




        self.amp *= self.sup
        self.amp = self.amp/tf.math.reduce_sum(self.amp)     

        self.shp1=phs.shape[:4]
        # qs = tf.ones((self.shp1[0],self.shp1[1],self.shp1[2],self.shp1[3],3))*self.Qs
        # qs = tf.math.round(self.Qs[tf.newaxis,:,:],3)
        
        # print(self.Qs.shape)
        # print( phs.shape)
        self.u = tf.linalg.lstsq( self.Qs, phs, l2_regularizer=0.0, fast=True, name=None)
        norm_L = tf.reduce_sum((tf.matmul(self.Qs,self.u)-phs)**2,axis=3)



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
        amp = tf.cast(amp/tf.reduce_sum(amp),dtype=tf.complex64)

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
                    r._cImage = ai.check_get_conj_reflect(self.recons[0]._cImage,r._cImage)

        
            plot_amp=False       
            if self.center:   
                if i%5 == 0:
                    
                    self.center_phase()  

            if i == len(recipes)-1:
                plot_amp = False
            self.phase_to_u(recipe[1],plot_amp)
            self.UpdateSupport()
            self.u_to_phase()
            
                  
            


        return


   
