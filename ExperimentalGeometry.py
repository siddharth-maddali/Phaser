# Defines modules that resolve the scattering geometry of a 
# Bragg coherent diffraction imaging experiment from a crystalline 
# sample, performed at Beamline 34-ID of the Advanced Photon 
# Source. Determines sampling bases of real and reciprocal 
# space, given a fixed experimental configuration, which includes 
# scattering direction, detector pixel size and orientation,
# beam energy and sample-detector distance.
#
# Siddharth Maddali
# Argonne National Laboratory
# August 2018


#	APS convention: 
#	    downstream: +Z
#	    outboard:   +X
#	    upward:     +y
#	Warning: 34-ID convention is different from this.
#       This script follows the more general APS convention.
#
#    Update: The paper in which we've described this transformation 
#    has been accepted for publication in the Journal of Applied 
#    Crystallography, and a nearly identical preprint can be found here: 
#    https://arxiv.org/abs/1909.05353
#

import numpy as np
import Misc as misc

class ScatteringGeometry:
   
    def __init__( self, 
        lam=1.3785e-10,                      # wavelength, meters, corresponding to 9 keV
        pix=55.e-6,                         # detector pixel size, meters
        arm=0.65,                           # sample-detector distance, meters
        gamma=45.,                          # degrees
        delta=25.,                          # degrees
        dtheta=0.01,                        # degrees (this becomes the step in phi if delta = 0).
        best_orthogonality=False,           # leave as False unless you know what you're doing!
        rocking_motor='th',               # Options are 'th', 'ph'. 
        recipSpaceSteps=[ 256, 256, 70 ]    # data array size

    ):
        """Initialize object with default values with:
                sg = ScatteringGeometry()

            or with any set of values you want, eg:
                sg = ScatteringGeometry( lam=2.64e-10, arm=0.6, peak=[ 2, 0, 0 ], a0=2.32e-10 )

            All the parameters you don't specify explicitly will be initialized to default values.

            'rocking_motor' manually sets the rocking direction: theta motor if 'th' and phi motor if 'ph'. 
            Only works if 'best_orthogonality' is set to False. 

            The 'best_orthogonality' parameter, when True, computes the mutual orthogonality parameter 
            separately for a theta rocking and a phi rocking, and chooses the better one. 
            Doesn't apply to experimental data, where the sampling is decided during the measurement, 
            but helps to simulate more well-behaved Fourier volumes in BCDI. Don't set to True if 
            you're reconstructing scans whose experimental geometry is already known. 
            If False, a rocking direction of 'th' is assumed unless specified otherwise. 
        """

        rocking_dict = { 'th':misc.Delta, 'ph':misc.Gamma }

        self._lambda = lam
        self._pix = pix
        self._arm = arm
        self._delta = delta * np.pi / 180.
        self._gamma = gamma * np.pi / 180.

        self._ki = ( 1. / self._lambda ) * np.array( [ 0., 0., 1. ] ).reshape( -1, 1 )     # incident unit vector  (3x1)    
        self._kf = misc.Delta( self._delta ) @ misc.Gamma( self._gamma ) @ self._ki         # scattered wave vector (3x1)
        self._Q = self._kf - self._ki
        self.detectorFrame = misc.Delta( self._delta ) @ misc.Gamma( self._gamma )

        detXY = self._pix / ( self._lambda * self._arm ) * self.detectorFrame @ np.array( [ [ 1., 0., 0. ], [ 0., 1., 0. ] ] ).T
            # the columns of this matrix are the reciprocal space steps in the detector x and y directions.

        self._dtheta = dtheta * np.pi / 180.
        self._recip = np.array( recipSpaceSteps ).astype( float ).reshape( 1, -1 )
        Brange = np.diag( 1. / self._recip.ravel() )

        if best_orthogonality==False: 

            self._Mtheta = rocking_dict[ rocking_motor ]
                # this is the 3x1 direction of changing Q.
                # Also, the NEGATIVE of the detector translation step.
            self._dq = ( self._Mtheta( self._dtheta ) @ self._Q ) - self._Q
            self._Brecip = np.concatenate( ( detXY, -1.*self._dq ), axis=1 )
                # The columns of this matrix are the sampling steps in 3 independent directions in reciprocal space.
                # Two of these are perpendicular (i.e. detector plane) while the third is the rocking direction.

        else: # used in simulations to choose the best orthogonal sampling of signal 
            _dq = [ ( _Mtheta( self._dtheta ) @ self._Q ) - self._Q   for _Mtheta in [ misc.Gamma, misc.Delta ] ]
            _Brecip = [ np.concatenate( ( detXY, -1.*q  ), axis=1 ) for q in _dq ]
            _ortho = [ np.absolute( misc.orthogonality( Bq ) ) for Bq in _Brecip ]
            _best = np.where( _ortho==max( _ortho ) )[0][0]
            self._dq = _dq[ _best ]
            self._Brecip = _Brecip[ _best ]



        self._Breal = np.linalg.inv( self._Brecip ).T @ Brange
            # The columns of this matrix are the sampling directions in 3D real space.

        self._Brecip    *= 1.e-9    # scaling to nm^-1 units
        self._Breal     *= 1.e+9    # scaling to nm units
        self.ortho      = [ misc.orthogonality( M ) for M in self.getSamplingBases() ]
            # these two numbers should be as close to 1 as possible  for better simulations. 
        return


    def getSamplingBases( self ):
        """getSamplingBases():
        returns a tuple containing the sampling bases in real space (nm) and reciprocal space (nm^-1)
        """
        return self._Breal, self._Brecip






            






