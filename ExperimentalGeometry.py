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


import numpy as np
import Misc as misc

class ScatteringGeometry:
   
    def __init__( self, 
        lam=1.3785e-10,                      # wavelength, meters, corresponding to 9 keV
        pix=55.e-6,                         # detector pixel size, meters
        arm=0.65,                           # sample-detector distance, meters
        a0= 4.078e-10,                      # lattice constant, meters
        peak= [ 1, 1, 1 ],                  # Bragg peak of interest
        gamma=45.,                          # degrees
        delta=25.,                          # degrees
        dtheta=0.01,                        # degrees (this becomes the step in phi if delta = 0).
        recipSpaceSteps=[ 256, 256, 70 ]    # data array size

    ):
        """Initialize object with default values with:
                sg = ScatteringGeometry()

            or with any set of values you want, eg:
                sg = ScatteringGeometry( lam=2.64e-10, arm=0.6, peak=[ 2, 0, 0 ], a0=2.32e-10 )

            All the parameters you don't specify explicitly will be initialized to default values.
        """

        self._lambda = lam
        self._pix = pix
        self._arm = arm
        self._a0 = a0
        self._peak = np.array( peak ).astype( float )
        self._delta = delta * np.pi / 180.
        self._gamma = gamma * np.pi / 180.

        self._ki = np.array( [ 0., 0., 1. ] ).reshape( -1, 1 )                              # incident unit vector  (3x1)    
        self._kf = misc.Delta( self._delta ) @ misc.Gamma( self._gamma ) @ self._ki         # scattered wave vector (3x1)
        self._Ghat = self._kf - self._ki
        self._Ghkl = ( self._Ghat / np.linalg.norm( self._Ghat ) ) * ( 
            2. * np.pi * np.linalg.norm( self._peak ) / self._a0
        ) # this is the 3x1 reciprocal lattice vector Ghkl around whith the diffraction pattern is centered


        self._dtheta = dtheta * np.pi / 180.
        self._recip = np.array( recipSpaceSteps ).reshape( 1, -1 )
        self._theta = np.arcsin( self._lambda * np.linalg.norm( self._peak ) / ( 2. * self._a0 ) )
        self._twotheta = 2. * self._theta


        if self._delta == 0.:   # scattering in the straight-up direction; phi motor is being rocked
            self._Mtheta = misc.Gamma
        else:                   # scattering off to the side; theta motor is being rocked
            self._Mtheta = misc.Delta

        self._dq = ( self._Mtheta( self._dtheta ) @ self._Ghkl ) - self._Ghkl   
            # this is the 3x1 step in reciprocal space along the rocking curve

        detXY = 2. * np.pi * self._pix / ( self._lambda * self._arm ) *\
            misc.Delta( self._delta ) @\
            misc.Gamma( self._gamma ) @\
            np.array( [ [ 1., 0., 0. ], [ 0., 1., 0. ] ] ).T
            # the columns of this matrix are the reciprocal space steps in the detector x and y directions.

        self._Brecip = np.concatenate( ( detXY, self._dq ), axis=1 ) 
            # The columns of this matrix are the sampling steps in 3 independent directions in reciprocal space.
            # Two of these are perpendicular (i.e. detector plane) while the third is the rocking direction.

        Brange = np.diag( 1. / self._recip.ravel() )

        self._Breal = 2. * np.pi *\
            np.linalg.inv( self._Brecip ).T @ Brange
            # The columns of this matrix are the sampling directions in 3D real space.

        self._Brecip    *= 1.e-9    # scaling to nm^-1 units
        self._Breal     *= 1.e+9    # scaling to nm units

    def getSamplingBases( self ):
        """getSamplingBases():
        returns a tuple containing the sampling bases in real space (nm) and reciprocal space (nm^-1)
        """
        return self._Breal, self._Brecip






            






