from score import score
import numpy as np
from coupled_phaser import cpr
import time    
import align_images as ai
from shrinkwrap2 import Shrinkwrap
from miscellaneous import *
import gc
import tensorflow as tf
from config import Config

import matplotlib.pyplot as plt
def run_ga(config_file):  
    
    
    """
        Function which runs the coupled phaser GA
    
    inputs:
        data -- hdf5 data file generated in interpolate notebook
        a0 -- lattice parameter (dtype=float units: nm)
        Recipes -- list of recipes for each generation (see Tutorial.ipynb for more info)
        num_gen -- number of generations in GA (dtype=int)
        num_ind -- number of individuals in first generation (dtype=int)
        cull -- list of factors by which to cull the population after each generation (list, dtype=int)
        criterion -- criterion by which to rank individuals for breeding (string, 'chi', 'sharp','max_volume','norm_sharp','least_squares')
        verbose -- option to print intermediate results (dtype=boolean)
        center -- option to center phases of each constituent about zero (dtype=boolean)
        pcc -- option to turn on partial coherence correction (dtype=boolean)
        gpu -- option to use gpu or not (dtype=boolean)
        unwrap_gens -- boolean list to turn on phase unwrapping generation by generation (list length: num_gen, dtype=boolean)
        plot_axis -- axis to cross-section for plotting (dtype=int)
        
    returns:
        vals -- dictionary containing the reconstruction object, list of chi^2 errors, and list of least squares losses after each generation
        
        """
    Cfg = Config(config_file)

    data,qs = extract_data(Cfg.data_file,Cfg.crop)
        

    if Cfg.indices is not None:
        qs = np.stack([qs[i] for i in Cfg.indices])
        data = np.stack([data[i] for i in Cfg.indices])
    
    
    if Cfg.unwrap_gens == None:
        unwrap_gens = np.zeros(Cfg.num_gen,dtype=np.bool)
        
    Cfg.cull = Cfg.cull[1:]
    initial =[[None,True] for i in range(Cfg.num_ind)]
    best_chi,best_L = [],[]
    tm_est = 0
    for g in range(Cfg.num_gen):
        unwrap = Cfg.unwrap_gens[g]
        recipe = Cfg.recipes[g]
        
        if Cfg.verbose:
            print('##########################################################################################')
            print('Generation: %s'%g)
        objs,chis,Ls = [],[],[] #lists which will contain the values for each individual
        for n in range(Cfg.num_ind):
            gc.collect()
            print('Individual: %s'%n)
            start = time.time()
            obj,rs = initial[n]
            
            #create multi-phaser object
            recon = cpr(data,qs,Cfg.a0,obj=obj,random_start=rs,pcc=Cfg.pcc,unwrap=unwrap,center=Cfg.center,gpu=Cfg.gpu) 
            recon.run_recipes(recipe)
            
            chi,L = recon.extract_error()
            obj = recon.extract_obj()
            
            tm = time.time()-start

            # add values for individual
            
            Ls.append(L[-1])
            objs.append(obj)
            
            chis.append(chi[-1])
            
            if Cfg.verbose:
                print('Time to Reconstruct: %s seconds'%np.round(tm,2))
                print('L:',L[-1],'\nChi:',chi[-1],'\n')

        chis = np.stack(chis)
        Ls = np.stack(Ls)
        objs = np.stack(objs)
        nans = np.isnan(chis)
        Ls[nans] = Ls.max()
        
        chis[nans] = chis.max()
        
        objs[nans] = obj[~nans][0]
        #dictionary containing inputs necessary for calculating each fitness metric in score.py
        w_dict = { 
                 'chi':chis,
                 'sharp':objs,
                 'max_volume':objs,
                 'norm_sharp':objs,
                 'least_squares':Ls} 
        
        scores,ws,winner = score(w_dict[Cfg.criterion],weight=Cfg.criterion)
        
        # store error values to determine whether error is converging
        best_chi.append(chis[winner])
        best_L.append(Ls[winner])
        
        m = max([a.shape[0] for a in objs])
        objs = [np.pad(a,([0,0],[(m-a.shape[0])//2,(m-a.shape[0])//2],[(m-a.shape[0])//2,(m-a.shape[0])//2],[(m-a.shape[0])//2,(m-a.shape[0])//2])) for a in objs]

        
        #breeding step before culling. align images first
        
        
        objs = [ai.check_get_conj_reflect_us(objs[winner],obj) for obj in objs]
        new_objs = [np.sqrt(objs[winner]*obj) for obj in objs]
        new_sups = [Shrinkwrap(np.absolute(obj)[0],1.0,0.1).numpy() for obj in objs]
        
        new_objs = [o*s for o,s in zip(new_objs,new_sups)]
            

            
        #culling step
        if g != Cfg.num_gen-1:
            if Cfg.cull[g] != 1:


                new_objs = [new_objs[n] for n in range(Cfg.num_ind) if scores[n]>Cfg.num_ind//Cfg.cull[g]]
                Cfg.num_ind = Cfg.num_ind//Cfg.cull[g]
        
        print('Individual %s Wins'%winner)
        shp2 = np.array(new_objs[1].shape[1:])
        
        shp1 = np.array(data[0].shape)
        
        paddings = (shp1-shp2)//2
        
        paddings = [[p,p] for p in paddings]
        new_objs = [np.pad(n,[[0,0],paddings[0],paddings[1],paddings[2]]) for n in new_objs]
        
        
        #set values for next generation
        

        initial = [[new_objs[n],False] for n in range(Cfg.num_ind)]
        
        if Cfg.verbose:
            
            m = plot_u(np.angle(objs[winner]),axis=Cfg.plot_axis)


    return {"obj":objs[winner],"chi":best_chi,"L":best_L}

