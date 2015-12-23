
# uncompress simple point LUT rawfile generated in matlab with two pass snappy/zip compression
# this is just for debug so compressed .sz.zip LUTs can be loaded back into matlab

import numpy as np
import os, sys
import snappy
import zipfile
from scipy import io as scio

#connFG = 6; connBG = 26;
connFG = 26; connBG = 6;

fn = 'simpleLUT3d_%dconnFG_%dconnBG' % (connFG,connBG)

zf = zipfile.ZipFile(fn + '.raw.sz.zip', mode='r'); simpleLUT = zf.read(fn + '.raw.sz'); zf.close()
simpleLUT = np.fromstring(snappy.uncompress(simpleLUT), dtype=np.uint8)
assert( simpleLUT.size == 2**27 )

pfn = os.path.join('tmp',fn)
simpleLUT.tofile(pfn + '.raw')
scio.savemat(pfn + '.mat',{'simpleLUT':simpleLUT})

