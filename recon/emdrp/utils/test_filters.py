
import numpy as np
from scipy import ndimage as nd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from Kuwahara import Kuwahara
from KuwaharaFourier import KuwaharaFourier

import time

I=mpimg.imread('/usr/local/MATLAB/R2016a/toolbox/images/imdata/cameraman.tif')
Id=mpimg.imread('/usr/local/MATLAB/R2016a/toolbox/images/imdata/cameraman.tif').astype(np.double)/255

t = time.time();
I2 = nd.gaussian_filter(I, [5.0, 5.0], mode='constant');
print('gaussian in %.4f s' % (time.time() - t))
print(I2.dtype, I2.max(), I2.min())

t = time.time();
I3 = nd.maximum_filter(I, [9, 9], mode='constant');
print('max in %.4f s' % (time.time() - t))
print(I3.dtype, I3.max(), I3.min())

t = time.time();
for i in range(256): I4 = Kuwahara(I, 5);
print('Kuwahara in %.4f s' % (time.time() - t))
print(I4.dtype, I4.max(), I4.min())

t = time.time();
for i in range(256): I5 = KuwaharaFourier(I, 5);
print('KuwaharaFourier in %.4f s' % (time.time() - t))
print(I5.dtype, I5.max(), I5.min())

plt.subplot(2,2,1); plt.imshow(I, cmap=plt.cm.gray, interpolation='nearest');
#plt.subplot(2,2,2); plt.imshow(I2, cmap=plt.cm.gray, interpolation='nearest');
plt.subplot(2,2,3); plt.imshow(I3, cmap=plt.cm.gray, interpolation='nearest');
plt.subplot(2,2,4); plt.imshow(I4, cmap=plt.cm.gray, interpolation='nearest');
plt.subplot(2,2,2); plt.imshow(I5, cmap=plt.cm.gray, interpolation='nearest');

#plt.imshow(I4, cmap=plt.cm.gray, interpolation='nearest');

plt.show()
