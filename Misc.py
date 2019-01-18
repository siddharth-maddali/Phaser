import numpy as np


def Gamma( gam ):
    return np.array( 
        [
            [ 1., 0., 0. ], 
            [ 0.,  np.cos( gam ), np.sin( gam ) ], 
            [ 0., -np.sin( gam ), np.cos( gam ) ]
        ]
    )

def Delta( delt ):
    return np.array( 
        [ 
            [  np.cos( delt ), 0., np.sin( delt ) ], 
            [ 0., 1., 0. ], 
            [ -np.sin( delt ), 0., np.cos( delt )]
        ]
    )

def Chi( chi ):
    return np.array( 
        [
            [  np.cos( chi ), -np.sin( chi ), 0. ], 
            [  np.sin( chi ),  np.cos( chi ), 0. ],
            [ 0., 0., 1. ] 
        ]
    )
