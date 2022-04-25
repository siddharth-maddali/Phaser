from score import score
import numpy as np
from coupled_phaser import cpr
import time    
import align_images as ai
from shrinkwrap2 import Shrinkwrap
from miscellaneous import plot_u,unpack_obj
import gc
import tensorflow as tf

import matplotlib.pyplot as plt
def run_ga(data,qs,sup,a,Recipes,num_gen,num_ind,cull,
           criterion = 'chi',
           verbose = False,
           pcc = False,
           gpu = 0,
           unwrap_gens = None,
           center = False,
           free_vox_mask = None,
           plot_axis = 2):  
    
    """
        Function which runs the coupled phaser GA
    
    inputs:
        data -- stack of datasets (numpy.ndarray, dtype=float,dimensions=(num_datasetsxnxnxn))
        qs -- q vectors for reflections corresponding to the datasets in data (nump.ndarray, dtype=float,dimensions=(num_datasetsx3)) 
        sup -- guess for object support (nump.ndarray, dtype=float,dimensions=(nxnxn)) 
        a -- lattice parameter (dtype=float units: nm)
        Recipes -- list of recipes for each generation (see Tutorial.ipynb for more info)
        num_gen -- number of generations in GA (dtype=int)
        num_ind -- number of individuals in first generation (dtype=int)
        cull -- list of factors by which to cull the population after each generation (list, dtype=int)
        criterion -- criterion by which to rank individuals for breeding (string, 'chi', 'sharp','max_volume','norm_sharp','least_squares')
        verbose -- option to print intermediate results (dtype=boolean)
        center -- option to center phases of each constituent about zero (dtype=boolean)
        pcc -- option to turn on partial coherence correction (dtype=boolean)
        free_vox_mask -- numpy mask to determine which voxels are used for optimization (nump.ndarray,dtype=boolean, dimensions = (nxnxn))
        gpu -- option to use gpu or not (dtype=boolean)
        unwrap_gens -- boolean list to turn on phase unwrapping generation by generation (list length: num_gen, dtype=boolean)
        plot_axis -- axis to cross-section for plotting (dtype=int)
        
    returns:
        vals -- dictionary containing u,amp,sup, list of chi^2 errors and list of least squares losses after each generation
        
        """
    
    
    
    if unwrap_gens == None:
        unwrap_gens = np.zeros(num_gen,dtype=np.bool)
        
    cull = cull[1:]
    initial =[[None,sup,True] for i in range(num_ind)]
    best_chi,best_L = [],[]
    tm_est = 0
    for g in range(num_gen):
        unwrap = unwrap_gens[g]
        recipes = Recipes[g]
        
        if verbose:
            print('##########################################################################################')
            print('Generation: %s'%g)
        objs,chis,Ls = [],[],[] #lists which will contain the values for each individual
        for n in range(num_ind):
            gc.collect()
            print('Individual: %s'%n)
            start = time.time()
            obj,sup,rs = initial[n]
            
            #create multi-phaser object
            recon = cpr(data,qs,a,obj=obj,random_start=rs,pcc=pcc,unwrap=unwrap,center=center,free_vox_mask=free_vox_mask) 
            recon.run_recipes(recipes)
            
            chi,L = recon.extract_error()
            obj = recon.extract_obj()
            
            tm = time.time()-start

            # add values for individual
            
            Ls.append(L[-1])
            objs.append(obj)
            
            chis.append(chi[-1])
            
            if verbose:
                print('Time to Reconstruct: %s seconds'%np.round(tm,2))
                print('L:',L[-1],'\nChi:',chi[-1],'\n')

        
        #dictionary containing inputs necessary for calculating each fitness metric in score.py
        w_dict = { 
                 'chi':chis,
                 'sharp':objs,
                 'max_volume':objs,
                 'norm_sharp':objs,
                 'least_squares':Ls} 
        
        scores,ws,winner = score(w_dict[criterion],weight=criterion)
        
        # store error values to determine whether error is converging
        best_chi.append(chis[winner])
        best_L.append(Ls[winner])
        
        m = max([a.shape[0] for a in objs])
        objs = [np.pad(a,([0,0],[(m-a.shape[0])//2,(m-a.shape[0])//2],[(m-a.shape[0])//2,(m-a.shape[0])//2],[(m-a.shape[0])//2,(m-a.shape[0])//2])) for a in objs]

        
        #breeding step before culling. align images first
        
        
        objs = [ai.check_get_conj_reflect_us(objs[winner],obj) for obj in objs]
        new_objs = [np.sqrt(objs[winner]*obj) for obj in objs]
        new_sups = [Shrinkwrap(np.absolute(obj)[0],1.0,0.1).numpy() for obj in objs]
#         amps = [np.absolute(img[0]) for img in imgs]
#         sups = [Shrinkwrap(amp,1.0,0.1).numpy() for amp in amps]
#         us = [np.angle(imgs[i])*sups[i] for i in range(len(imgs))]
#         new_amps = [np.sqrt(amps[winner]*amp) for amp in amps]
#         new_sups = [Shrinkwrap(amp,1.0,0.1).numpy() for amp in new_amps]
#         new_us = [np.mean([us[i],us[winner]],axis=0)*new_sups[i] for i in range(len(us))]
            

            
        #culling step
        if g != num_gen-1:
            if cull[g] != 1:

#                 new_amps = [new_amps[n] for n in range(num_ind) if scores[n]>num_ind//cull[g]]
                new_sups = [new_sups[n] for n in range(num_ind) if scores[n]>num_ind//cull[g]]
                new_objs = [new_objs[n] for n in range(num_ind) if scores[n]>num_ind//cull[g]]
                num_ind = num_ind//cull[g]
        print('Individual %s Wins'%winner)
        shp2 = np.array(new_sups[0].shape)
        
        shp1 = np.array(data[0].shape)
        
        paddings = (shp1-shp2)//2
        
        paddings = [[p,p] for p in paddings]
        new_objs = [np.pad(n,[[0,0],paddings[0],paddings[1],paddings[2]]) for n in new_objs]
        new_sups = [np.pad(n,paddings) for n in new_sups]
        
        #set values for next generation
        initial = [[new_objs[n],new_sups[n],False] for n in range(num_ind)]
        
        if verbose:
            
            plot_u(np.angle(objs[winner]),axis=plot_axis)


    return {"obj":objs[winner],"chi":best_chi,"L":best_L}

