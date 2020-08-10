##############################################################
#
#	GAFFT:
#            Geometry-aware far-field propagators customized 
#            for Bragg CDI. The propagators work between 
#            orthogonal grids in the real-space of the scatterer 
#            and the sheared space of the far-field diffraction 
#            pattern. Contains both forward and backward propagators.
#            
#            Reference: https://doi.org/10.1107/S1600576720001375
#
#	    Siddharth Maddali
#	    Argonne National Laboratory
#	    August 2020
#           smaddali@alumni.cmu.edu
#
##############################################################


import numpy as np

try: 
    from pyfftw.interfaces.numpy_fft import fftshift, fftn, ifftn
except:
    from numpy.fft import fftshift, fftn, ifftn


class Mixin: 

    def setupDiffractionGeometry( self ):

        self._grid = [ fftshift( this ) for this in np.meshgrid( *[ np.arange( -n//2., n//2. ) for n in self._support.shape ] ) ]
        self._grid[1] = np.flip( self.grid[1], axis=0 )

        dq_tilde = np.sqrt( ( self._Bq**2 ).sum( axis=0 ) )
        self._Bq = self._Bq / dq_tilde.reshape( 1, -1 ).repeat( 3, axis=0 )
        N1, N2, N3 = self._grid[0].shape
        al, bt, gm = tuple( Bq[:,-1] ) # detector-frame direction cosines of sampling vector along rocking direction
        dq3 = gm * dq_tilde[2]
        
        dr3 = 1./ ( gm * N3 * dq_tilde[2] )
        dr2 = 1./ ( dq_tilde[1] * np.ceil( ( N2*dq_tilde[1] + bt*N3*dq_tilde[2] )/( dq_tilde[1] ) ) )
        dr1 = 1./ ( dq_tilde[0] * np.ceil( ( N1*dq_tilde[0] + bt*N3*dq_tilde[2] )/( dq_tilde[0] ) ) )

        self.realSpaceSteps = dr1, dr2, dr3

        self.__mu__     = np.exp( -1.j * 2. * np.pi * ( dq3*zs * ( (al/gm)*dr1*xs + (bt/gm)*dr2*ys ) ) )
        self.__mustar__ = np.conj( self.__mu__ ) # one used for forward propagation, the other for backward propagation.

        self._myfftn = gafftn         # new geometry-aware propagators
        self._myifftn = gaifftn

        return
    

    def gafftn( self, arr ):
        arr_out = fftn( self.__mu__*fftn( arr, axes=[2] ), axes=[ 0, 1 ] )
        return arr_out

    def gaifftn( self, arr ):
        arr_out = fftn( self.
        return arr_out
