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

def orthogonality( M ):
    """
    Computes the mutual orthogonality 𝒪(M) of the column vectors of a a 3x3 masis matrix M. 
    𝒪(M) ∈ [−1,1], where a larger |𝒪(M)| means the basis vectors are "more mutually orthogonal" 
    than for a smaller |𝒪(M)|. |𝒪(M)| = 1 implies perfect mutual orthogonality. The sign of 𝒪(M) 
    indicates the handedness of the basis vectors with respect to a right-handed system. 
    """
    return np.linalg.det( M ) / np.prod( np.sqrt( ( M**2 ).sum( axis=0 ) ) )
