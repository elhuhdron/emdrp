
from czifile import CziFile
#from utils import showImgData
import numpy as np
#from scipy.misc import imresize

#import matplotlib as plt
from matplotlib import pylab as pl

#fn='/home/pwatkins/Downloads/0.08 lead_continuous 150-50-50-03.czi'
fn = '/data/pwatkins/kara/sample_alignment_data_from_Kara_20180125/2017/training-zebrafish-4-sections/' +\
        'zebrafish_20171013_10-15-10/003_Region3.czi'

czi = CziFile(fn)
#print(czi.metadata, dir(czi.metadata))
iimg = czi.asarray()
print(iimg.shape)

#shape = np.array(iimg.shape[3:5])
#img = np.zeros(shape//10 + 1,dtype=iimg.dtype)
#
#for scene in range(iimg.shape[1]):
#    cimg = np.squeeze(iimg[0,scene,0,:,:,0])[0::10,0::10]
#    sel = (cimg > 0)
#    img[sel] = cimg[sel]

img = np.squeeze(iimg)

interp_string = 'nearest'
pl.figure(1);
pl.imshow(img,interpolation=interp_string, cmap='gray');
#pl.title(i)
#pl.colorbar()
pl.show()
