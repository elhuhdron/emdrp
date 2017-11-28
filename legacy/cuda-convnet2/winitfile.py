
import os
import numpy as n
import numpy.random as nr
from python_util.gpumodel import IGPUModel

def loadw(name, idx, shape, params=None):
    fn = os.path.join(params[0], params[1])
    rows, cols = shape
    return n.fromfile(fn, dtype=n.single).reshape(rows,cols)

def loadb(name, shape, params=None):
    fn = os.path.join(params[0], params[1])
    rows, cols = shape
    return n.fromfile(fn, dtype=n.single).reshape(rows,cols)

def loadwcp(name, idx, shape, params=None):
    load_dic = IGPUModel.load_checkpoint(params[0])
    rows, cols = shape
    return load_dic['model_state']['layers'][params[1]]['weights'][idx].reshape(rows,cols)

def loadbcp(name, shape, params=None):
    load_dic = IGPUModel.load_checkpoint(params[0])
    rows, cols = shape
    return load_dic['model_state']['layers'][params[1]]['biases'].reshape(rows,cols)

def makew(name, idx, shape, params=None):
    stdev, mean = float(params[0]), float(params[1])
    rows, cols = shape
    return n.array(mean + stdev * nr.randn(rows, cols), dtype=n.single)

def makeb(name, shape, params=None):
    stdev, mean = float(params[0]), float(params[1])
    rows, cols = shape
    return n.array(mean + stdev * nr.randn(rows, cols), dtype=n.single)

def makewIdent(name, idx, shape, params=None):
    chans = int(params[0]); dscale = float(params[1]); rows, cols = shape
    filtPix = rows/chans; side = int(n.sqrt(filtPix)); center = side/2; 
    #print chans,filtPix,side,center
    a = n.zeros((chans,side,side,cols), dtype=n.single); a[:,center,center,:] = 1.0/dscale
    return a.reshape((rows,cols))

