import tensorflow as tf
import numpy as np

from numpy.fft import fftshift, fftn, ifftn

mu = 1.
sigma = 0.5
photon_max = 5.e4


data = Namespace( **sio.loadmat( '/home/smaddali/ANL/simulatedCrystals/crystals/crystal_3.mat' ) )
intens = fftshift( data.intens )

grid = np.meshgrid( *[ np.arange( -n//2., n//2. ) for n in intens.shape ] )
pts = np.concatenate( tuple( fftshift( this ).reshape( 1, -1 ) for this in grid ), axis=0 )
D = mu + 0.5*sigma*( -1. + 2.*np.random.rand( 3 ) )
R = np.linalg.svd( np.random.rand( 3, 3 ) )[0]
C = R @ np.diag( 1. / D )**2 @ R.T
g = np.exp( -0.5 * ( pts * ( C @ pts ) ).sum( axis=0 ).reshape( grid[0].shape ) ) / ( ( 2.*np.pi )**( 3./2. ) * np.prod( D ) )

intens_pc = np.absolute( ifftn( fftn( intens ) * fftn( g ) ) )
signal_pc = np.random.poisson( photon_max * intens_pc / intens_pc.max() )
mod = fftshift( signal_pc )
