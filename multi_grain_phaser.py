import numpy as np
import matplotlib.pyplot as plt
import h5py
import coupled_phaser as cpr
from miscellaneous import *
import gc
from align_twins import align_grains
import time
import tensorflow as tf 


class multi_grain_phaser:
    def __init__(self,a0,peaks_file_1,o_1,peaks_file_2,o_2,u_vol_1,u_vol_2,
                 inds_1 = None,inds_2 = None,crop=None):
        self.a0 = a0
        self.o_1,self.o_2 = o_1,o_2
        self.data_1,self.gs_1 = extract_data(peaks_file_1,crop)
        
        self.data_2,self.gs_2 = extract_data(peaks_file_2,crop)
        self.grain_1_u_vol = u_vol_1
        self.grain_2_u_vol = u_vol_2
        if inds_1 is not None:
            self.gs_1 = np.stack([self.gs_1[i] for i in inds_1])
            self.data_1 = np.stack([self.data_1[i] for i in inds_1])
        if inds_2 is not None:
            self.gs_2 = np.stack([self.gs_2[i] for i in inds_2])
            self.data_2 = np.stack([self.data_2[i] for i in inds_2])

            
        
        
        return
    

    def run_recipe(self,recipe_1,unwrap_gens,param_gens,recipe_2 = None,guess_2 = None,gpu_device=0):
        if recipe_2 is None:
            if guess_2 is None:
                recipe_2 = recipe_1

        grain_1_params = None
        grain_2_params = None
        grain_1 = None
        grain_2 = None
        self.err = []
        shp = np.array(self.data_1.shape)//2
        for i in range(len(recipe_1)):
            print(i)
            start = time.time()
            rs=True if i==0 else False
            

            with tf.device('/GPU:%s'%gpu_device):
                recon_1 = cpr.cpr(self.data_1,self.gs_1,self.a0,obj=grain_1,
                                  random_start=rs,pcc=True,unwrap=unwrap_gens[i],params = grain_1_params)
                recon_1.run_recipes(recipe_1[i])
                grain_1 = recon_1.extract_obj()
                
                
                
                
                if guess_2 is not None:
                    grain_2 = guess_2.copy()
                else:
                    recon_2 = cpr.cpr(self.data_2,self.gs_2,self.a0,obj=grain_2,
                                      random_start=rs,pcc=True,unwrap=unwrap_gens[i],params = grain_2_params)
                    recon_2.run_recipes(recipe_2[i])
                    grain_2 = recon_2.extract_obj()
                

                if param_gens[i]:
                    grain_1_params = recon_1.extract_params()
                    if guess_2 is None:
                        grain_2_params = recon_2.extract_params()
                else: 
                    grain_1_params = None
                    grain_2_params = None
                self.err = self.err+list(np.array(recon_1.extract_error()[0]))

                ################# Unpack and add in u_vol  #################
                grain_1_u,grain_1_amp,grain_1_sup = unpack_obj(grain_1)
                grain_1_u = grain_1_u+self.grain_1_u_vol*grain_1_sup


                grain_2_u,grain_2_amp,grain_2_sup = unpack_obj(grain_2)
                grain_2_u = grain_2_u+self.grain_2_u_vol*grain_2_sup

                ################# Transform to Sample Frame ##################
                grain_1_u_samp = np.reshape(self.o_1@grain_1_u.reshape((3,-1)),grain_1_u.shape)
                grain_2_u_samp = np.reshape(self.o_2@grain_2_u.reshape((3,-1)),grain_2_u.shape)
                grain_1_samp = grain_1_amp*np.exp(1j*grain_1_u_samp)
                grain_2_samp = grain_2_amp*np.exp(1j*grain_2_u_samp)

                ####################### Align Grains ##########################

                u,amp,sup1,sup2 = align_grains(grain_1_samp,grain_2_samp,match_u=True,coupled=False)
            if i%5 == 0:
                plot_u(u)
                fig,ax = plt.subplots(ncols=2,figsize=(7,3))
                ax[0].imshow((sup1+sup2*2)[20:-20,20:-20,shp[2]].T,origin='lower')
                ax[1].imshow(amp[20:-20,20:-20,shp[2]].T,origin='lower')
                plt.show()

            ###################### Update Guesses ##########################
            grain_1_u = u*sup1

            grain_1_u = np.reshape(self.o_1.T@grain_1_u.reshape((3,-1)),grain_1_u.shape)
            grain_1_u = (grain_1_u - self.grain_1_u_vol)*sup1

            grain_2_u = u*sup2
            grain_2_u = np.reshape(self.o_2.T@grain_2_u.reshape((3,-1)),grain_2_u.shape)
            grain_2_u = (grain_2_u - self.grain_2_u_vol)*sup2



            grain_1_amp = np.float64(sup1)*amp
            grain_2_amp = np.float64(sup2)*amp


            self.grain_1 = sup1*grain_1_amp*np.exp(1j*grain_1_u)
            self.grain_2 = sup2*grain_2_amp*np.exp(1j*grain_2_u)



            gc.collect()
            print(np.round(time.time()-start),'seconds')
            print('######################################')
        return
    
    def extract_objs(self):
        return self.grain_1,self.grain_2
    def extract_err(self):
        return self.err
        