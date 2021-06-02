
# compress simple point LUT rawfile generated in matlab with two pass snappy/zip compression

import numpy as np
import os, sys
import snappy
import zipfile

#connFG = 6; connBG = 26;
connFG = 26; connBG = 6;

fn = 'simpleLUT3d_%dconnFG_%dconnBG.raw' % (connFG,connBG)

f = open(os.path.join('tmp',fn), 'rb'); simpleLUT = np.fromfile(f, dtype=np.uint8, count=2**27); f.close()

with open(fn + '.sz', 'wb') as fh: fh.write(snappy.compress(simpleLUT.tostring()))
zf = zipfile.ZipFile(fn + '.sz.zip', mode='w')
zf.write(fn + '.sz', compress_type=zipfile.ZIP_DEFLATED); zf.close()

if os.path.isfile(fn + '.sz'): os.remove(fn + '.sz')


