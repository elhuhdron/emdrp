# original from matlab central:
#  https://www.mathworks.com/matlabcentral/fileexchange/58260-super-fast-kuwahara-image-filter--for-n-dimensional-real-or-complex-data-
# translated to python watkinspv 20 Dec 2016
# only translated portion for fast 2d using convolution (small kernels)
# xxx - this turned out to be slower in python with doFourier=False, must have to do with complex convolution???
#   need winsize of about 9 or greater for doFourier=True to perform better than normal fast Kuwahara
def KuwaharaFourier(original, winsize, doFourier=True):
    #function filtered = KuwaharaFourier(original,winsize)
    #% Edge aware Kuwahara filter for 2D and 3D images:
    #% - black and white only (scalar values)
    #% - not suitable for 2D colored image (eg. RGB)
    #% Author Job G. Bouwman: jgbouwman@hotmail.com

    import numpy as np
    #from scipy import ndimage as nd
    from scipy import signal as signal
    from scipy import fftpack as fft

    inshape = list(original.shape); wcenter = (winsize+1)//2

    #%% Incorrect input handling
    if np.issubdtype(original.dtype, np.integer):
        dtype = np.single
    else:
        dtype = original.dtype
    test=np.zeros((2,2),dtype=dtype)
    if type(original) != type(test):
        raise Exception( 'In Kuwahara, original is not *NumPy* array')
    if len(inshape) != 2:
        raise Exception( 'In Kuwahara, original is not 2 dimensional')
    if original.dtype != dtype:
        #raise Exception( 'In Kuwahara, source not correct data type')
        original = original.astype(dtype)

    # wrong-sized kernel is an error
    if winsize % 2 == 0:
        raise Exception( 'In Kuwahara, window size not odd')

    #%% Build the square convolution kernel
    N_or            = inshape;
    Dimensionality  = 2;
    Nsquare         = wcenter;
    Nkernel         = np.ones((Dimensionality,),dtype=np.int64)*Nsquare;
    Nrem            = winsize - Nsquare;
    squareKernel    = 1/Nsquare**Dimensionality*np.ones(Nkernel, dtype=dtype);
    #%% Convolution-wise calculation of averages and variances
    #% NB: I have accelerated the convolution, by processing the averages and
    #% the variances in a single convolution. This can be done by adding the
    #% squared image as a an imaginary passenger to the normal image.
    complexConvInput = original + 1j*original**2;
    if not doFourier:
        #% Convolute...
        #% ... (in case of small kernels) in the spatial domain:
        #complexConvOutput = nd.filters.convolve(complexConvInput, squareKernel, mode=mode);
        complexConvOutput = signal.convolve(complexConvInput, squareKernel);
    else:
        #% ... (in case of large kernels) in the Fourier domain:
        #% NB: Zero-padding and ensuring equal size of image and kernel
        complexConvInput = np.pad(complexConvInput, [[0,x] for x in Nkernel], mode='constant');
        squareKernel     = np.pad(squareKernel    , [[0,x] for x in N_or],    mode='constant');
        #% Multiplication in frequency domain:
        complexConvOutput = fft.ifftn(fft.fftn(complexConvInput)*fft.fftn(squareKernel));

    #%% Separate the real and imaginary part:
    #% Real part is the running average:
    runningAvrge = np.real(complexConvOutput);
    #% imag part is the stds, only squared average must be yet subtracted:
    runningStdev = np.imag(complexConvOutput)-runningAvrge**2;

    #%% Now select from the four quadrants (the eight octants) the right value
    #% - first create stack of four (eight) images
    #% - select minimal stdev
    #% - take the corresponding avarage

    #% Patch the North-East side, North-West side, ... onto a stack:
    avg_Stack = np.stack((\
        runningAvrge[Nrem:Nrem+N_or[0], Nrem:Nrem+N_or[1]],
        runningAvrge[Nrem:Nrem+N_or[0],     :N_or[1]],
        runningAvrge[    :N_or[0],      Nrem:Nrem+N_or[1]],
        runningAvrge[    :N_or[0],          :N_or[1]]),
        axis=2);
    std_Stack = np.stack((\
        runningStdev[Nrem:Nrem+N_or[0], Nrem:Nrem+N_or[1]],
        runningStdev[Nrem:Nrem+N_or[0],     :N_or[1]],
        runningStdev[    :N_or[0],      Nrem:Nrem+N_or[1]],
        runningStdev[    :N_or[0],          :N_or[1]]),
        axis=2);

    #% Choice of the index with minimum variance
    indices = std_Stack.argmin(axis=2); ##ok<ASGLU>

    #% Selecting the accompanying average value:
    x,y = np.indices(inshape)
    return avg_Stack[x,y,indices]
