#!/usr/bin/env python

# The MIT License (MIT)
#
# Copyright (c) 2016 Paul Watkins, National Institutes of Health / NINDS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


#import os
import argparse
import time
import numpy as np
import cv2

from dpLoadh5 import dpLoadh5
from dpWriteh5 import dpWriteh5

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*20, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

class dpWarp(dpWriteh5):

    def __init__(self, args):
        dpWriteh5.__init__(self,args)

        # print out all initialized variables in verbose mode
        if self.dpWarp_verbose: print('dpWarp, verbose mode:\n'); print(vars(self))

    def loadData(self):
        if self.dpWarp_verbose:
            print('Loading data'); t = time.time()

        # probability data is main data cube (srcfile)
        self.readCubeToBuffers(); probdata = self.data_cube

        # load the raw em data (rawfile)
        #loadh5 = dpLoadh5.readData(srcfile=self.rawfile, dataset=self.raw_dataset, chunk=self.chunk.tolist(),
        #    offset=self.offset.tolist(), size=self.size.tolist(), verbose=self.dpLoadh5_verbose)
        #rawdata = loadh5.data_cube

        #self.data_to_warp = 255 - (rawdata*(1-probdata)).astype(np.uint8)  # matt's method
        self.data_to_warp = (255*(1-probdata)).astype(np.uint8)

        # free space incase of large volumes
        self.data_cube = np.zeros((0,), dtype=self.data_cube.dtype)

        if self.dpWarp_verbose:
            print('\tdone in %.4f s' % (time.time() - t, ))

    def warp_probs(self):


        data = self.data_to_warp; assert( data.dtype == np.uint8 )
        prevgray = data[:,:,0]
        show_figures = False
        self.warps = np.zeros(self.size.tolist() + [2],dtype=np.float32)

        if self.dpWarp_verbose:
            print('Warping %d slices' % (self.size[2]-1,)); t = time.time()
        for z in range(1,self.size[2]):
            #if self.dpWarp_verbose:
            #    print('Warping to slice %d / %d' % (z,self.size[2])); t = time.time()

            gray = data[:,:,z]
            #flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.8, 8, 16, 10, 7, 1.5, 0)
            self.warps[:,:,z,:] = flow

            if show_figures:
                #cv2.imshow('flow', draw_flow(gray, flow))
                cv2.imshow('prevgray',prevgray); cv2.imshow('gray',gray)
                cv2.imshow('flow HSV', draw_hsv(flow))
                warpedgray = warp_flow(prevgray, flow)
                cv2.imshow('warped', warpedgray)
                cv2.imshow('diff', np.abs(gray-warpedgray))
                cv2.waitKey(0)

            prevgray = gray
            #if self.dpWarp_verbose:
            #    print('\tdone in %.4f s' % (time.time() - t, ))
        if show_figures: cv2.destroyAllWindows()

        if self.dpWarp_verbose:
            print('\tdone in %.4f s' % (time.time() - t, ))

    def writeWarps(self):
        self.data_cube = self.warps[:,:,:,0]; self.dataset = 'warpx'; self.writeCube()
        self.data_cube = self.warps[:,:,:,1]; self.dataset = 'warpy'; self.writeCube()

    @staticmethod
    def addArgs(p):
        dpWriteh5.addArgs(p)
        # adds arguments required for this object to specified ArgumentParser object
        p.add_argument('--rawfile', nargs=1, type=str, default='raw.h5', help='Path/name of hdf5 raw EM (input) file')
        p.add_argument('--raw-dataset', nargs=1, type=str, default='data', help='Name of the raw EM dataset to read')
        p.add_argument('--dpWarp-verbose', action='store_true', help='Debugging output for dpWarp')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(\
        description='Extends hdf5 data volume to create dense warping along resliced z direction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    dpWarp.addArgs(parser)
    args = parser.parse_args()

    wrp = dpWarp(args)
    wrp.loadData()
    wrp.warp_probs()
    wrp.writeWarps()

