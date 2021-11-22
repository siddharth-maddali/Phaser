import numpy as np
from scipy.spatial.transform import Rotation



def Gamma( gam ):
    return Rotation.from_rotvec( 
        gam * np.array( [ -1., 0., 0. ] ) 
    ).as_matrix()

def Delta( delt ):
    return Rotation.from_rotvec( 
        delt * np.array( [ 0., 1., 0. ] )
    ).as_matrix()

def Chi( chi ):
    return Rotation.from_rotvec( 
        chi * np.array( [  0., 0., 1. ] )
    ).as_matrix()

def orthogonality( M ):
    """
    Computes the mutual orthogonality 𝒪(M) of the column vectors of a a 3x3 masis matrix M. 
    𝒪(M) ∈ [−1,1], where a larger |𝒪(M)| means the basis vectors are "more mutually orthogonal" 
    than for a smaller |𝒪(M)|. |𝒪(M)| = 1 implies perfect mutual orthogonality. The sign of 𝒪(M) 
    indicates the handedness of the basis vectors with respect to a right-handed system. 
    """
    return np.linalg.det( M ) / np.prod( np.sqrt( ( M**2 ).sum( axis=0 ) ) )
