import numpy as np

# These metrics taken from Andrew's paper

def Chi( mod_inferred, mod_measured ):
    return np.nansum( ( mod_inferred - mod_measured )**2 )

def Sharp( mod_inferred, mod_measured ):
    return  np.nansum( mod_inferred**4 )

def SharpNorm( mod_inferred, mod_measured ):
    return np.nansum( mod_inferred**( 0.25 ) )

def MaxVolume( mod_inferred, mod_measured ):
    return -np.nansum( mod_inferred ) / mod_inferred.max()
