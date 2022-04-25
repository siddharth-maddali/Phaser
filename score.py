import numpy as np
from scipy.stats import rankdata as rk
from miscellaneous import unpack_obj


def score(args,verbose=False,weight='poisson_log'):
    
    if weight=='sharp':
        amps = [unpack_obj[1] for obj in args]
        sharp = np.array([np.sum(np.absolute(amp)**4) for amp in amps])
        weights = sharp/np.sum(sharp)
        weights = [1/w for w in weights]
        weights = list(np.array(weights)/sum(weights))
    elif weight =='norm_sharp':
        amps = [unpack_obj[1] for obj in args]
        norm_sharp = np.array([np.sum(np.absolute(amp)**0.25)**4 for amp in amps])
        weights = norm_sharp/np.sum(norm_sharp)
        weights = [1/w for w in weights]
        weights = np.array(weights)/sum(weights)
        
    elif weight == 'max_volume':
        sups = [unpack_obj[2] for obj in sups]
        max_volume = np.array([np.sum(sup) for sup in sups])
        weights = max_volume/np.sum(max_volume)
    else:
        weights = np.array(args)/sum(args)
        weights = [1/w for w in weights]
        weights = list(np.array(weights)/sum(weights))
        

        
    scores = rk(weights)
    
    winner = scores.argmax()
    if verbose:
        print('individual %s wins'%winner)
    return scores,weights,winner
    
    




    

