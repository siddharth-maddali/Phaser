import numpy as np
from scipy.stats import rankdata as rk



def score(args,verbose=False,weight='poisson_log'):
    
    if weight=='sharp':
        sharp = np.array([np.sum(np.absolute(amp)**4) for amp in args])
        weights = sharp/np.sum(sharp)
        weights = [1/w for w in weights]
        weights = list(np.array(weights)/sum(weights))
    elif weight =='norm_sharp':
        norm_sharp = np.array([np.sum(np.absolute(amp)**0.25)**4 for amp in args])
        weights = norm_sharp/np.sum(norm_sharp)
        weights = [1/w for w in weights]
        weights = np.array(weights)/sum(weights)
        
    elif weight == 'max_volume':
        max_volume = np.array([np.sum(sup) for sup in args])
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
    
    




    

    