from score import score
import numpy as np
from coupled_phaser import cpr
import time    
import align_images as ai
from shrinkwrap2 import Shrinkwrap
from miscellaneous import plot_u
import gc

import matplotlib.pyplot as plt
def run_ga(data,qs,sup,a,Recipes,num_gen,num_ind,cull,criterion='chi',verbose=False,pcc=False,gpu=True,unwrap_gens=None):  
    
    """
        Function which runs the coupled 
    
    inputs:
        data -- stack of datasets (numpy array, dtype=float,dimensions=(num_datasetsxnxnxn))
        qs -- q vectors for reflections corresponding to the datasets in data (numpy array, dtype=float,dimensions=(num_datasetsx3)) 
        sup -- guess for object support (numpy array, dtype=float,dimensions=(nxnxn)) 
        a -- lattice parameter (dtype=float units: nm)
        Recipes -- list of recipes for each generation (see Tutorial.ipynb for more info)
        num_gen -- number of generations in GA (dtype=int)
        num_ind -- number of individuals in first generation (dtype=int)
        cull -- list of factors by which to cull the population after each generation (list, dtype=int)
        criterion -- criterion by which to rank individuals for breeding (string, 'chi', 'sharp','max_volume','norm_sharp','least_squares')
        verbose -- option to print intermediate results (dtype=boolean)
        center -- option to phases of each constituent about zero (dtype=boolean)
        pcc -- option to turn on partial coherence correction (dtype=boolean)
        free_vox_mask -- numpy mask to determine which voxels are used for optimization (numpy array,dtype=boolean, dimensions = (nxnxn))
        gpu -- option to use gpu or not (dtype=boolean)
        unwrap_gens -- boolean list to turn on phase unwrapping generation by generation (list length: num_gen, dtype=boolean)
        
    returns:
        vals -- dictionary containing u,amp,sup, list of chi^2 errors and list of least squares losses after each generation
        
        """
    
    
    
    if unwrap_gens == None:
        unwrap_gens = np.zeros(num_gen,dtype=np.bool)
        
    
    initial =[[sup,sup,np.zeros((3,sup.shape[0],sup.shape[1],sup.shape[2])),True] for i in range(num_ind)]
    best_chi,best_L = [],[]
    tm_est = 0
    for g in range(num_gen):
        unwrap = unwrap_gens[g]
        gc.collect()
        recipes = Recipes[g]
        if verbose:
            print('##########################################################################################')
            print('Generation: %s'%g)
        us,amps,sups,chis,Ls = [],[],[],[],[] #lists which will contain the values for each individual
        for n in range(num_ind):
            gc.collect()
            print('Individual: %s'%n)
            start = time.time()
            amp,sup,u,rs = initial[n]
            obj = cpr(data,qs,sup,a,amp=amp,u=u,random_start=rs,pcc=pcc,gpu=gpu,unwrap=unwrap) #create multi-phaser object
            obj.run_recipes(recipes)
            chi,L = obj.extract_error()
            vals = obj.extract_vals()
            u = vals['u']
            amp = vals['amp']
            sup = vals['sup']
            tm = time.time()-start
#             gc.collect()
            # add values for individual
            
            Ls.append(L[-1])
            amps.append(amp)
            sups.append(sup)
            us.append(u)
            
            chis.append(chi[-1])
            
            if verbose:
                print('Time to Reconstruct: %s seconds'%np.round(tm,2))
                print('L:',L[-1],'\nChi:',chi[-1],'\n')




        #dictionary containing inputs necessary for calculating each fitness metric in score.py
        w_dict = { 
                 'chi':chis,
                 'sharp':amps,
                 'max_volume':sups,
                 'norm_sharp':amps,
                 'least_squares':Ls} 
        
        scores,ws,winner = score(w_dict[criterion],weight=criterion)
        
        # store error values to determine whether error is converging
        best_chi.append(chis[winner])
        best_L.append(Ls[winner])
        
        m = max([a.shape[0] for a in amps])
        amps = [np.pad(a,([(m-a.shape[0])//2,(m-a.shape[0])//2],[(m-a.shape[0])//2,(m-a.shape[0])//2],[(m-a.shape[0])//2,(m-a.shape[0])//2])) for a in amps]
        sups = [np.pad(a,([(m-a.shape[0])//2,(m-a.shape[0])//2],[(m-a.shape[0])//2,(m-a.shape[0])//2],[(m-a.shape[0])//2,(m-a.shape[0])//2])) for a in sups]
        
        us = [np.pad(a,([0,0],[(m-a.shape[1])//2,(m-a.shape[1])//2],[(m-a.shape[1])//2,(m-a.shape[1])//2],[(m-a.shape[1])//2,(m-a.shape[1])//2])) for a in us]
        
        #breeding step before culling. align images first
        
        imgs = [amps[i]*np.exp(us[i]*1j) for i in range(len(us))]
        imgs = [ai.check_get_conj_reflect_us(imgs[winner],img) for img in imgs]
        amps = [np.absolute(img[0]) for img in imgs]
        sups = [Shrinkwrap(amp,1.0,0.1).numpy() for amp in amps]
        us = [np.angle(imgs[i])*sups[i] for i in range(len(imgs))]
        new_amps = [np.sqrt(amps[winner]*amp) for amp in amps]
        new_sups = [Shrinkwrap(amp,1.0,0.1).numpy() for amp in new_amps]
        new_us = [np.mean([us[i],us[winner]],axis=0)*new_sups[i] for i in range(len(us))]
            
#         #breeding during and after culling step. No image alignment
#         if g >= cull[0]:
#             new_amps = [np.sqrt(amps[winner]*amp) for amp in amps]
#             new_sups = [Shrinkwrap(amp,1.0,0.1).numpy() for amp in new_amps]
#             new_us = [np.mean([us[i],us[winner]],axis=0)*new_sups[i] for i in range(len(us))]
            
        #culling step
        if cull[g] != 1:
            
            new_amps = [new_amps[n] for n in range(num_ind) if scores[n]>num_ind//cull[g]]
            new_sups = [new_sups[n] for n in range(num_ind) if scores[n]>num_ind//cull[g]]
            new_us = [new_us[n] for n in range(num_ind) if scores[n]>num_ind//cull[g]]
            num_ind = num_ind//cull[g]
        print('Individual %s Wins'%winner)
        shp2 = np.array(new_amps[0].shape)
        
        shp1 = np.array(data[0].shape)
        
        paddings = (shp1-shp2)//2
        
        paddings = [[p,p] for p in paddings]
        new_amps = [np.pad(n,paddings) for n in new_amps]
        new_sups = [np.pad(n,paddings) for n in new_sups]
        new_us = [np.pad(n,[[0,0],paddings[0],paddings[1],paddings[2]]) for n in new_us]
        #set values for next generation
        initial = [[new_amps[n],new_sups[n],new_us[n],False] for n in range(num_ind)]
        
        if verbose:
            plot_u(us[winner],axis=2)
#             fig,axs = plt.subplots(nrows=1,ncols=3,figsize= (15,5))
#             c = np.array(us[winner].shape)[1]//2
#             labels = ['$u_x$','$u_y$','$u_z$']
#             for i,ax in enumerate(axs):
#                 s = us[winner][i,c-5:c+5,c-5:c+5,c-5:c+5]
#                 minn,maxx = s.mean()-3*np.std(s),s.mean()+3*np.std(s)
#                 A = ax.imshow(us[winner][i,:,:,c],vmin=minn,vmax=maxx)
#                 ax.set_title(labels[i])
#                 fig.colorbar(A,ax=ax)

#             plt.show()

    return {"amp":amps[winner],"sup":Shrinkwrap(amps[winner],1.0,0.2),"u":us[winner],"error":best_chi,"L":best_L}