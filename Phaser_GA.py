import copy
import sys
from score import score
from matplotlib.colors import LogNorm
import align_images as ai
from shrinkwrap import Shrinkwrap as sw
from numpy.fft import fftshift,fftn

import numpy as np
import Phaser as ph
import matplotlib.pyplot as plt
def run_GA(signal,recipes,num_gen,num_ind,cull,support=None,fitness='chi',pcc=False,plot=False):
    
    """Runs a genetic algorithm by calling Phaser several times in serial. 
    
    Inputs:
        signal - 3D data array (npy)
        recipes - a list of recipe strings to be used for each generation of the GA, len(recipes) = num_gen (list)
        support - initial support guess. If none, initial support will be generated in Phaser (npy)
        num_gen - number of generations for the GA (int)
        num_ind - number of individuals in the initial population (int)
        cull_gen - generation at which the population will be culled, reducing the number of individuals by 50% (int)
        fitness - metric by which the fittest individual will be determined ('chi','sharp','norm_sharp','max_volume')
        pcc - option to turn on partial coherence correction (boolean)
        plot - option to plot reconstruction at the end of each generation (boolean)
        
    Returns:
        final_img - complex array for real-space reconstructed object (npy)
        final_sup - final support from reconstructed object (npy)
        track_gen_err - list of chi values for the fittest individual at the end of each generation (npy)
        track_err - array of chi values after each iteration over all generations (npy)
        """
    
    
    
    
    if len(recipes)==1:
        recipes = [recipes[0] for i in range(num_gen)]
    new_sups =[support for i in range(num_ind)]
    new_imgs = [support for i in range(num_ind)]
    track_gen_err = []
    track_err = []
    for g in range(num_gen):
        print('GENERATION %s'%g)
        imgs,sups,chis,errors = [],[],[],[]
        
        for i in range(num_ind):
            print('Individual %s'%(i))
            
            rs = True if g==0 else False
            
            recon = ph.Phaser( modulus=np.sqrt(signal),
                              support = new_sups[i],
                              random_start=rs,
                              img_guess = new_imgs[i],
                              pcc=pcc,
                              gpu=True ).gpusolver
            recon.runRecipe( recipes[g], show_progress=False )
            recon.Retrieve()
#             blur_kernels.append(recon._pccSolver._blurKernel.numpy())
            chi = recon._error[-1]
            
            img = recon.finalImage
            sup = recon.finalSupport
            phase = np.angle(img)*sup
            amp = np.absolute(img)/np.absolute(img).max()

            supp = sw(amp,1.0,0.5)
            avg = np.mean(phase[np.where(supp!=0)])
            phase -= avg
            phase*=sup         

            imgs.append(amp*np.exp(phase*1j))
            sups.append(sup)
            chis.append(chi)
            errors.append(recon._error)
        w_dict = {
             'chi':chis,
             'sharp':[np.absolute(img) for img in imgs],
             'max_volume':sups,
             'norm_sharp':[np.absolute(img) for img in imgs]}
        scores,ws,winner = score(w_dict[fitness],weight=fitness)
        
        imgs = [ai.check_get_conj_reflect(imgs[winner],img) for img in imgs]
        
        new_imgs = [np.sqrt(imgs[winner]*img) for img in imgs]
        new_sups = [sw(np.absolute(img),1.0,0.1) for img in new_imgs]
        
        
        if plot:
            aa = np.absolute(imgs[winner])
            fig,ax = plt.subplots(ncols = 3,figsize=(15,5))
            A = ax[0].imshow(signal[:,:,imgs[winner].shape[2]//2],norm=LogNorm()) 
            ax[0].set_title('Signal from Data')
            fig.colorbar(A,ax=ax[0])
            B=ax[1].imshow(np.absolute(fftshift(fftn(fftshift(imgs[winner])))**2)[:,:,imgs[winner].shape[2]//2],norm=LogNorm())
            ax[1].set_title('Forward-Modeled Signal')
            fig.colorbar(B,ax=ax[1])
            C=ax[2].imshow((aa/aa.max())[:,:,imgs[winner].shape[2]//2],vmax=1,vmin=0)
            ax[2].set_title('Reconstructed Amplitude')
            fig.colorbar(C,ax=ax[2])
            plt.show()
            
            
        track_err.append(errors[winner])
        track_gen_err.append(chis[winner])
        if cull[g] != 1:
            new_amps = [new_imgs[n] for n in range(num_ind) if scores[n]>num_ind//cull[g]]
            new_sups = [new_sups[n] for n in range(num_ind) if scores[n]>num_ind//cull[g]]
            num_ind = num_ind//cull[g]
            
        final_img = imgs[winner]
        print('individual %s wins'%winner)
        final_sup = sw(np.absolute(final_img),1.0,0.1)
    return final_img,final_sup,track_gen_err,np.hstack(track_err)#,blur_kernels[winner]

