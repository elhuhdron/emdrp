# original from matlab central:
#  https://www.mathworks.com/matlabcentral/fileexchange/15027-faster-kuwahara-filter
# translated to python watkinspv 19 Dec 2016
def Kuwahara(original, winsize, mode='constant'):
    #function filtered = Kuwahara(original,winsize)
    #Kuwahara   filters an image using the Kuwahara filter
    #   filtered = Kuwahara(original,windowSize) filters the original image with a
    #                                            given windowSize and yields the result in filtered
    #
    # This function is optimised using vectorialisation, convolution and
    # the fact that, for every subregion
    #     variance = (mean of squares) - (square of mean).
    # A nested-for loop approach is still used in the final part as it is more
    # readable, a commented-out, fully vectorialised version is provided as
    # well.
    #
    # This function is about 2.3 times faster than KuwaharaFast at
    # http://www.mathworks.com/matlabcentral/fileexchange/loadFile.do?objectId=13474&objectType=file
    # with a 5x5 window, and even faster at higher window sizes (about 4 times on a 13x13 window)
    #
    # Inputs:
    # original      -->    image to be filtered
    # windowSize    -->    size of the Kuwahara filter window: legal values are
    #                      5, 9, 13, ... = (4*k+1)
    #
    # Example
    # filtered = Kuwahara(original,5);
    #
    # Filter description:
    # The Kuwahara filter works on a window divided into 4 overlapping
    # subwindows (for a 5x5 pixels example, see below). In each subwindow, the mean and
    # variance are computed. The output value (located at the center of the
    # window) is set to the mean of the subwindow with the smallest variance.
    #
    #    ( a  a  ab   b  b)
    #    ( a  a  ab   b  b)
    #    (ac ac abcd bd bd)
    #    ( c  c  cd   d  d)
    #    ( c  c  cd   d  d)
    #
    # References:
    # http://www.ph.tn.tudelft.nl/DIPlib/docs/FIP.pdf
    # % http://www.incx.nec.co.jp/imap-vision/library/wouter/kuwahara.html
    #
    # Copyright Luca Balbi, 2007

    import numpy as np
    from scipy import ndimage as nd

    inshape = list(original.shape); inshapek = inshape + [4]; wcenter = (winsize+1)//2; wcenter2 = wcenter**2

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

    #%% Build the subwindows
    tmpavgker = np.zeros((winsize,winsize),dtype=dtype);
    tmpavgker[:wcenter, :wcenter] = 1.0/wcenter2;

    # tmpavgker is a 'north-west' subwindow (marked as 'a' above)
    # we build a vector of convolution kernels for computing average and
    # variance
    avgker = [None]*4
    avgker[0] = tmpavgker;                 # North-west (a)
    avgker[1] = np.fliplr(tmpavgker);      # North-east (b)
    avgker[3] = np.flipud(tmpavgker);      # South-east (c)
    avgker[2] = np.fliplr(avgker[3]);      # South-west (d)

    # this is the (pixel-by-pixel) square of the original image
    squaredImg = original**2;

    #%% Calculation of averages and variances on subwindows
    avgs = np.zeros(inshapek, dtype=dtype);
    stddevs = np.zeros(inshapek, dtype=dtype);
    for k in range(4):
        avgs[:,:,k] = nd.filters.correlate(original,avgker[k], mode=mode);      # mean on subwindow
        stddevs[:,:,k] = nd.filters.correlate(squaredImg,avgker[k], mode=mode); # mean of squares on subwindow
        stddevs[:,:,k] = stddevs[:,:,k]-avgs[:,:,k]**2;                         # variance on subwindow

    #%% Choice of the index with minimum variance
    indices = stddevs.argmin(axis=2); ##ok<ASGLU>

    #%% Take mean subwindows corresponding to min variance subwindows
    x,y = np.indices(inshape)
    return avgs[x,y,indices]
