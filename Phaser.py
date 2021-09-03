"""
    Basic Python module for phasing Bragg CDI data

	    Siddharth Maddali
	    Argonne National Laboratory
	    2017-2018
                                                                               
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@                           %@     %@     *@     #@     /@     *@     #@     ,@     ,@     #@      @    &@@@@@@@
@@@@@@@*     ,&@@(                  @.     @      @(     @.     @.     @(     @,     @/     @#     @,  (@@ *      @@@@@@
@@@@@      @@@@@@@@                #@     /@     ,@     (@     ,@     ,@     (@      @     .@     *@  &@@@ @@@      (@@@
        ,@@@@@@@@@@               ,@.     @.     @*    ,@.     @,     @*    .@.     @*     @*     @.  @@@ (@@@@@/       
      (@@@@@@@@@@@                %%     (&     (&     #%     /@     /@     (%     /@     /@     /&    @( @@@@@@@@%     
     @@@@@@@@@@@.        @@@%    *@      @      @     ,@      @      @     .@      @      @ ,@@@,@      @@@@@@@@@@@@/   
     /@@@@@@@/       &@@@@@@@@*  @/     @&     @&     @(     @@     @&     &#     @@     @@@@@@@..@%   @@  /@@@@@@@/    
                   @@@@@@@@@@@* .@      @      @     .@      @      @      @      @      @ @@@@@ @@@@@ @                
                &@@@@@@@@@@@@#  @#     @@     @@     &%     @@     @@     %&     @@     @@ .@@@* @@@@@  @               
               @@@@@@@@@@@@@    @ .%@@@ *     @      @      @      @      @.    * @@@&. @    %@ %@@@@@ @@@              
               @@@@@@@@@@      %.@@@@@  @@   @@     #&     &@     @@     (@    @. @@@@@.      /@@@@@@. @@@              
                 &&&&*       /@ @@@@@@ @@@(  @      @.     @      @      @,   #@ @@@@@@ @@@%  @*   (&.%&                
                           @@@* @@@@@. @@@  @@     /@     %@     %@     /@     . @@@@@/ @@@@@#&     %@                  
                         @@@@@ %@@@@@ &@    @@@@@@@       @.     @   @@@ (@&   @.@@@@@ &@@@@@ /@    @.                  
                        @@@@@# @@@@@,,    (, @@@@@# @@%  #@     /@@@@@@# @@@@@%@   #@% @@@@@% @@@  #@                   
                         @@@@ #@@@@*@.  @@@ @@@@@@ /@@@  @.     @,@@@@@ /@@@@@ %@    %(@@@@@ *@@   @,                   
                            *&     #@%@@@@( @@@@@% @@@. (@     .@  @@@% @@@@@/ @@@@& @     ,&     *@                    
                            @/    .,%@@@@@ &@@@@@ /@&  / %@@@@@ #*  %@ /@@@@@ %@@@@@ ,     @(     @,                    
                           #&     ,*@@@@@*,@@@@@*.%  @@,*@@@@@/.@@@@ #(@@@@@*,@@@@@/ @@   #&     (%                     
                           @.    * @@@@@@ &@@@@@&.,@@@% @@@@@@ &@@@@@ %  @@& &@@@@@ &@@   @.     @.                     
                          @&     @/#@@@@ .@@@  @.@@@@@ /@@@@@  @@@@@ .@@@  %*@@@@@  @%   @&     %%                      
                          @     .@     .&    &@ @@@@@& @@@@@@ @@@@@@ @@@@@@,    ..&      @      @                       
                         @&     @#     @@   @@  @@@@@.,@@@@@  @@@@@  @@@@@,.@    @@     @@     #@                       
                         @      @      @   @@@ @@@@@@ @@@@@@ @@@@@@ @@@@@@ &@@   @      @      @.                       
                        @@     %%     &@   %@  @@@@@*.@@@@@. @@@@@  @@@@@/ @@/  &@     &@     /@                        
                        @      @      @       @@@@@@ &@@@@@ @@@@@@ @@@@@@ %@    @      @      @*                        
                       @@     (@    %. @@@@@@@ @@@@( @@@@@. @@@@@. @@@@@(    @@, @@@@ (@     *@                         
                       @      @.  @@@ @@@@@@ @@#  @ #@@@@@ &@@@@@ @@@@@,@&@@@@@ &@@@@@ .     @/                         
                      @@     *@   @@, @@@@@, @@@@@*@#@@@@* @@@@@/ @@&  # @@@@@* @@@@@%      ,@                          
                      @      @*    @ &@@@@@ @@@@@@ /. .@@ %@@@@@ %   @@ /@@@@@ %@@@@@ /     @#                          
                     #@     ,@     ##@@@@@( @@@@@% @@@  (@/@@@/.@  @@@% @@@@@/ @@@@@/@     .@                           
                     @    @@ *@@@@ @,  @@@ %@@@@@ *@@@@ @*     @/ @@@@ *@@@@@ #@@@  @% @@@@ ,@@*                        
                    %(  @@@,.@@@@@.*@.   #,@@@@@*.@@@@&%#     #&  @@@*.@@@@@,*@    # @@@@@*.@@@@@                       
                    @.   @@ %@@@@% &@@@# @. *@@@ %@@@..@.     @,    @ #@@@@&#.   (@ #@@@@@ #@@@@.                       
                   @(      .@@@@@ /@@@@@       @&     @(     @@     @&     &#  %@@  @@@@@  @@@@                         
                   @      @ .@@@% @@@@@@ @(    @     .@      @      @      @  #@@@ @@@@@@ @@*                           
               @@@ .@@@@@@&    % *@@@@@  @(   @@     &%     @@     @@     %&  #@@  @@@@@        %@@@@@@@@@              
              #@@@ @@@@@@ @%    @ @@@@@ @     @      @      @      @      @.    @ @@@@@ @    *@@@@@@@@@@@@&             
                @  @@@@@  @@@, %%     &@     @@     #&     &@     @@     (@     &@     &@   @@@@@@@@@@@@@               
                 @@@@@@@ @@@@@/@      @      @      @.     @      @      @,     @      @   @@@@@@@@@@@@                 
      /@@@@@*   &@  ,@@  @@@@@@@     &@     @@     /@     %@     %@     /@     %@     (@   @@@@@@@@@.       *@@@@@%     
    @@@@@@@@@@@/@      ,@@@@@/@.     @      @      @*     @.     @      @/     @.     @.   .@@@@@#       ,@@@@@@@@@@@   
     .@@@@@@@@@, @    @@     *@     %@     %@     *@     #@     /@     *@     #@     ,@                @@@@@@@@@@@@(    
        &@@@@@@ &@@   @      @*     @.     @      @(     @.     @.     @(     @,     @/               &@@@@@@@@@@       
@@@/      (@@@* @@@  #@     ,@     #@     /@     ,@     (@     .@     .@     (@      @                @@@@@@@@/      *@@
@@@@@@#      @ %@@   @      @#     @,     @,     @#     @*     @/     @#     @*     @%                 @@@@@      ,@@@@@
@@@@@@@@/    #&     (&     *@     #&     *&     *@     (&     ,@     *@     /&     ,@                           .@@@@@@@


"""

import numpy as np
import GPUModule as accelerator

import os

try:
    from pyfftw.interfaces.numpy_fft import fftshift, fftn, ifftn
except: 
    from numpy.fft import fftshift, fftn, ifftn

# plugin modules
import Core, RecipeParser
import ER, HIO, SF
import GaussPCC, Morphology


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppresses all Tensorflow terminal messages

class Phaser( 
        Morphology.Mixin,       # Routines to manipulate object support
        GaussPCC.Mixin,         # New ER accounting for partial coherence
        Core.Mixin,             # core CPU algorithms and routines
        RecipeParser.Mixin,     # for handling recipe strings
        ER.Mixin,               # error reduction
        HIO.Mixin,              # hybrid input/output
        SF.Mixin                # solvent flipping
    ):

    def __init__( self,
            modulus,
            support=None,       # if None, use default support 
            beta=0.9, 
            binning=1,          # for high-energy CDI. Set to 1 for regular phase retrieval.
            gpu=False,
            pcc=False,
            pcc_params=None,    # user-defined coherence function parameters
            random_start=True 
            ):
       
        self.BEStruct           = np.ones( ( 3, 3, 3 ) ) # default structuring element for 3D binary erosion
        self.BinaryErosion      = self.__CPUErosion__
        
        self._modulus           = fftshift( modulus )
        self._arraySize         = tuple( this*shp for this, shp in zip( [ binning, binning, 1 ], self._modulus.shape ) )
        if support is None:
            self._initializeSupport()
        else: 
            self._support = fftshift( support )

        self._support_comp      = 1. - self._support
        self._beta              = beta

        if random_start:
            self._cImage            = np.exp( 2.j * np.pi * np.random.random_sample( self._arraySize ) ) * self._support
        else:
            self._cImage            = 1. * self._support

       
        self._cachedImage       = np.zeros( self._cImage.shape ).astype( complex )
        self._cImage_fft_mod = np.absolute( fftn( self._cImage ) )

        self._error             = []

        self._UpdateError()
        self.generateAlgoDict()

        if gpu==True:
            gpack = self.generateGPUPackage( pcc=pcc, pcc_params=pcc_params )
            self.gpusolver = accelerator.Solver( gpack )

        #if pcc==True:
        #    self._pccSolver = PCSolver( np.absolute( self._modulus )**2, gpack )
        #    self._kernel_f = self._pccSolver.getBlurKernel()
        #    self._ModProject = self._ModProjectPC

        return

