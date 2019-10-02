##############################################################
#
#	Core:
#           Contains all the core routines and algorithm 
#           implementations for CPU.
#
#	    Siddharth Maddali
#	    Argonne National Laboratory
#	    Oct 2019
#           smaddali@alumni.cmu.edu
#
##############################################################

class Mixin:

# Writer function to manually update support
    def UpdateSupport( self, support ):
        self._support = support
        self._support_comp = 1. - self._support
        return

# Writer function to manualy reset image
    def ImageRestart( self, cImg, reset_error=True ):
        self._cImage = cImg
        if reset_error:
            self._error = []
        return

# Reader function for the final computed modulus
    def Modulus( self ):
        return np.absolute( fftshift( fftn( self._cImage ) ) )

# Reader function for the error metric
    def Error( self ):
        return self._error

# Updating the error metric
    def _UpdateError( self ):
        self._error += [ 
            ( 
                ( self._cImage_fft_mod - self._modulus )**2 * self._modulus 
            ).sum() / self._modulus_sum 
        ]
        return

# Error reduction algorithm
    def ER( self, num_iterations, show_progress=False ):
        if show_progress:
            allIterations = tqdm( list( range( num_iterations ) ), desc=' ER' )
        else:
            allIterations = list( range( num_iterations ) )
        for i in allIterations:
            self._ModProject()
            self._cImage *= self._support
            self._cImage_fft_mod = np.absolute( fftn( self._cImage ) )
            self._UpdateError()
        return

# Hybrid input/output algorithm
    def HIO( self, num_iterations, show_progress=False ):
        if show_progress:
            allIterations = tqdm( list( range( num_iterations ) ), desc='HIO' )
        else:
            allIterations = list( range( num_iterations ) )
        for i in allIterations:
            origImage = self._cImage.copy() 
            self._ModProject()
            self._cImage = ( self._support * self._cImage ) +\
                self._support_comp * ( origImage - self._beta * self._cImage )
            self._UpdateError()
        return

# Solvent flipping algorithm
    def SF( self, num_iterations, show_progress=False ):
        if show_progress:
            allIterations = tqdm( list( range( num_iterations ) ), desc=' SF' )
        else:
            allIterations = list( range( num_iterations ) )
        for i in allIterations:
            self._ModHatProject()
            self._ModProject()
            self._UpdateError()
        return

# Basic shrinkwrap with gaussian blurring
    def Shrinkwrap( self, sigma, thresh ):
        result = gaussian_filter( 
            np.absolute( self._cImage ), 
            sigma, mode='constant', cval=0.
        )
        self._support = ( result > thresh*result.max() ).astype( float )
        self._support_comp = 1. - self._support
        return

# The projection operator into the modulus space of the FFT.
# This is a highly nonlinear operator.
    def _ModProject( self ):
        self._cImage = ifftn( 
            self._modulus * np.exp( 1j * np.angle( fftn( self._cImage ) ) ) 
        )
        return

# The reflection operator in the plane of the (linear)
# support operator. This operator is also linear.
    def _SupReflect( self ):
        self._cImage = 2.*( self._support * self._cImage ) - self._cImage
        return

# The projection operator into the 'mirror image' of the
# ModProject operator in the plane of the support projection
# operator. The involvement of the ModProject operator
# makes this also a highly nonlinear operator.
    def _ModHatProject( self ):
        self._SupReflect()
        self._ModProject()
        self._SupReflect()
        return

# The alignment oeprator that centers the object after phase retrieval.
    def Retrieve( self ):
        self.finalImage = self._cImage
        self.finalSupport = self._support
        self.finalImage, self.finalSupport = post.centerObject( 
            self.finalImage, self.finalSupport
        )
        return

# Generates a package for the GPU module to read and generate tensors.
    def generateGPUPackage( self ):
        mydict = { 
            'modulus':self._modulus, 
            'support':self._support, 
            'beta':self._beta, 
            'cImage':self._cImage
        }
        return mydict
