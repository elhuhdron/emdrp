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

#import os, sys
import numpy as np
from scipy import ndimage as nd
import scipy.sparse as sparse
from threading import Thread

from pyCext import binary_warping

def pix_fscore_metric( lbl_truth, lbl_proposed, calcAll=False ):
    # calcAll is to return all associated pixel metrics. false only returns those used by labrainth frontend

    n = lbl_truth.size

    # positive are values labeled as ICS (arbitary decision, but convention)
    nlbl_truth = np.logical_not(lbl_truth); nlbl_proposed = np.logical_not(lbl_proposed)
    tp = (np.logical_and(lbl_truth, lbl_proposed)).sum(dtype=np.int64)
    if calcAll: tn = (np.logical_and(nlbl_truth, nlbl_proposed)).sum(dtype=np.int64)
    fp = (np.logical_and(nlbl_truth, lbl_proposed)).sum(dtype=np.int64)
    fn = (np.logical_and(lbl_truth, nlbl_proposed)).sum(dtype=np.int64)

    # true positive rate (also recall), set to 1 if undefined (no positives in truth)
    P = (tp + fn).astype(np.double); tpr_recall = tp/P if P > 0 else 1.0

    # precision (also positive predictive value), set to 0 if undefined (no positives in proposed)
    PP = (tp + fp).astype(np.double); precision = tp/PP if PP > 0 else 0.0

    if calcAll:
        pixel_error = (fp + fn).astype(np.double) / n if n > 0 else 0.0
        # false positive rate, set to 0 if undefined (no negatives in truth)
        N = (fp + tn).astype(np.double); fpr = fp/N if N > 0 else 1.0

    # frontend refers to this as "accuracy" (not actually accuracy).
    # is actually F1 score which is harmonic mean of precision and recall.
    pr_sum = precision + tpr_recall; fScore = (2.0 * precision * tpr_recall / pr_sum) if (pr_sum > 0) else 0.0

    if calcAll:
        return fScore, tpr_recall, precision, pixel_error, fpr, tp, tn, fp, fn
    else:
        return fScore, tpr_recall, precision, tp, fp, fn

def make_consensus( votes, cnt, tie_include ):
    if (cnt % 2) or not tie_include:
        # odd number of users in consensus, no ties, must have majority
        # OR even number of users in consensus, tie excludes voxel from consensus
        return (votes > cnt/2)
    else:
        # even number of users in consensus, tie includes voxel in consensus
        return (votes >= cnt/2)

def pixel_error_fscore( lbl_truth, lbl_proposed ):
    return pix_fscore_metric(lbl_truth, lbl_proposed, calcAll=True)

def warping_error( lbl_truth, lbl_proposed, doComps=True, simpleLUT=None, connectivity=1 ):

    warped, classified, nonSimpleTypes, diff, simpleLUT = binary_warping( lbl_truth, lbl_proposed,
        return_nonSimple=True, connectivity=connectivity, simpleLUT=simpleLUT )

    wrp_err = float(diff) / lbl_truth.size

    if doComps:
        lbls, nSplits = nd.measurements.label(classified == nonSimpleTypes['dic']['RESULT_SPLIT'],
            structure=np.ones((3,3,3),dtype=np.bool))
        nonSimpleTypesOut = np.zeros(lbls.shape, dtype=lbls.dtype)
        nonSimpleTypesOut[lbls > 0] = nonSimpleTypes['dic']['RESULT_SPLIT']
        lbls, nMerges = nd.measurements.label(classified == nonSimpleTypes['dic']['RESULT_MERGE'],
            structure=np.ones((3,3,3),dtype=np.bool))
        nonSimpleTypesOut[lbls > 0] = nonSimpleTypes['dic']['RESULT_MERGE']
    else:
        nSplits = (classified == nonSimpleTypes['dic']['RESULT_SPLIT']).sum(dtype=np.int64)
        nMerges = (classified == nonSimpleTypes['dic']['RESULT_MERGE']).sum(dtype=np.int64)
        nonSimpleTypesOut = nonSimpleTypes

    return wrp_err, nSplits, nMerges, nonSimpleTypesOut, simpleLUT

# modified version of adapted rand error (ISBI 2013):
#   author: Ignacio Arganda-Carreras (iarganda@mit.edu)
#   More information at http://brainiac.mit.edu/SNEMI3D
#
# this version by default does not exclude the background for calculating the metric.
# use nogtbg to exclude ground truth background.
# calculation then only differs by a factor of -n in Pk, Qk, Tk.
#   unclear why this was omitted from the original ISBI version, but negligible for large n.
#
# function split into subroutines for creating the sparse matrix and calculating rand subroutines
#   for easier use with resampling.
def adapted_rand_error( lbl_truth, lbl_proposed, nogtbg = False, getRI=False, eri=0 ):
    return adapted_rand_error_confusion(*adapted_rand_error_getconfusion( lbl_truth, lbl_proposed, nogtbg ),
        getRI=getRI, eri=eri)

def adapted_rand_error_getconfusion( lbl_truth, lbl_proposed, nogtbg = False ):
    segT = np.ravel(lbl_truth); segP = np.ravel(lbl_proposed)
    n = segT.size; assert( n == segP.size ) # images must be same size

    nlabelsT = segT.max()+1; nlabelsP = segP.max()+1

    # variable names match definitions from Fowlkes and Mallows,
    # "A Method for Comparing Two Hierarchical Clusterings"

    # compute overlap matrix, AKA contingency or confusion matrix
    # xxx - this can probably be fixed to not go dense? is there a point?
    #m_ij = sparse.csr_matrix((np.ones(n, dtype=np.double), (segT, segP)), shape=(nlabelsT, nlabelsP))
    m_ij = sparse.csr_matrix((np.ones(n, dtype=np.double), (segT, segP)), shape=(nlabelsT, nlabelsP)).toarray()

    # optionally remove ground truth background from metric
    if nogtbg: m_ij = m_ij[1:,:]; n = (segT > 0).sum(dtype=np.int64)
    #assert( m_ij.sum(dtype=np.int64) == n )    # assert should be true, but comment for speed

    return m_ij, n

def adapted_rand_error_confusion( m_ij, n=None, getRI=False, eri=0 ):
    if n is None: n = m_ij.sum(dtype=np.int64)

    # sum over proposed labels in contingency matrix
    #m_idot = np.asarray(m_ij.sum(1))
    m_idot = m_ij.sum(1)

    # sum over ground truth labels in contingency matrix
    #m_dotj = np.asarray(m_ij.sum(0))
    m_dotj = m_ij.sum(0)

    # calculate values essentially equal to number of pairwise combinations
    Pk = np.sum(m_idot * m_idot) - n  # 2 * number of pairs together in ground truth
    Qk = np.sum(m_dotj * m_dotj) - n  # 2 * number of pairs together in proposed
    #Tk = np.sum(m_ij.toarray() ** 2)     - n  # 2 * number of pairs together in both
    Tk = np.sum(m_ij * m_ij)     - n  # 2 * number of pairs together in both

    # Rand index
    if getRI:
        ri = 1 - (0.5*(Pk + Qk) - Tk) / (n*(n-1)/2);
        # expected value of the rand index
        if eri < 0:
            eri = 1 - (Pk + Qk)/(n*(n-1)) + (2*Qk*Pk)/(n*n*(n-1)*(n-1));
        # adjusted rand index
        if eri != 0:
            ari = (ri - eri) / (1 - eri);

    # precision
    # pairs together in both out of pairs together in proposed
    prec = Tk / Qk;

    # recall
    # pairs together in both out of pairs together in ground truth
    rec = Tk / Pk;

    # F-score
    fScore = 2.0 * prec * rec / (prec + rec);

    # adapted rand error
    re = 1.0 - fScore;

    if getRI:
        # specify positive eri (E[RI]) to use as expected value.
        # specify negative eri to calculate it in the normal way (hypergeometric distribution) above
        # specify zero eri to not calculate adapted rand index
        if eri != 0:
            return re, prec, rec, ri, ari
        else:
            return re, prec, rec, ri
    else:
        return re, prec, rec

# for easy parallelization of rand point/object resampling
class adapted_rand_error_resample_objects_points_thread(Thread):
    def __init__(self, n, samps_per_thread, ares, precs, recs, selObj, inds, selPts, lbl_proposed):
        Thread.__init__(self)
        self.n = n; self.samps_per_thread = samps_per_thread
        self.ares = ares; self.precs = precs; self.recs = recs;
        self.selObj = selObj; self.inds = inds; self.selPts = selPts; self.lbl_proposed = lbl_proposed

    def run(self):
        #print(self.n*self.samps_per_thread,(self.n+1)*self.samps_per_thread)
        samples = range(self.n*self.samps_per_thread,(self.n+1)*self.samps_per_thread)
        for s in samples:
            # the resampled ground truth for this iteration
            retruth = np.zeros(self.lbl_proposed.size, self.lbl_proposed.dtype)
            # combine object and point selects to create resampled ground truth
            objs = np.nonzero(self.selObj[:,s])[0]
            for o in objs: retruth[self.inds[o][self.selPts[o][:,s]]] = o+1
            self.ares[s], self.precs[s], self.recs[s] = adapted_rand_error(retruth.reshape(self.lbl_proposed.shape),
                self.lbl_proposed, nogtbg=True)

def adapted_rand_error_resample_objects_points(lbl_truth, lbl_proposed, nObjects, nPoints, nSamples=100, pCI=0.05,
        getDistros=False, nThreads=1):
    #import time

    ntObjs = lbl_truth.max(); assert( lbl_truth.shape == lbl_proposed.shape )

    #print('resample'); t = time.time()

    # pre-generate selects for objects
    # bernoulli resampling of the objects with mean rate set to get mean of nObjects
    selObj = (np.random.rand(ntObjs,nSamples) < (nObjects / float(ntObjs)))
    # pre-generate selects for points for all objects
    inds = [None]*ntObjs; selPts = [None]*ntObjs; #nPts = [None]*ntObjs;
    for o in range(ntObjs):
        # bernoulli resampling of points within objects to get mean of nPoints per each object
        inds[o] = np.flatnonzero(lbl_truth == o+1)
        selPts[o] = (np.random.rand(inds[o].shape[0],nSamples) < (nPoints / float(inds[o].shape[0])))

    ares = np.zeros((nSamples,),dtype=np.double)
    precs = np.zeros((nSamples,),dtype=np.double)
    recs = np.zeros((nSamples,),dtype=np.double)

    '''
    for s in range(nSamples):
        retruth = np.zeros(lbl_truth.size, lbl_truth.dtype)     # the resampled ground truth for this iteration
        # combine object and point selects to create resampled ground truth
        objs = np.nonzero(selObj[:,s])[0]
        for o in objs: retruth[inds[o][selPts[o][:,s]]] = o+1
        ares[s], precs[s], recs[s] = adapted_rand_error(retruth.reshape(lbl_truth.shape), lbl_proposed, nogtbg=True)
    '''

    threads = nThreads * [None]
    assert( nSamples % nThreads == 0 ) # xxx - currently not dealing with remainders
    samps_per_thread = nSamples / nThreads
    for i in range(nThreads):
        threads[i] = adapted_rand_error_resample_objects_points_thread(i, samps_per_thread, ares, precs, recs,
            selObj, inds, selPts, lbl_proposed)
        threads[i].start()
    for i in range(nThreads): threads[i].join()

    #print('\tdone in %.4f s' % (time.time() - t))
    if getDistros:
        return ares, precs, recs
    else:
        return adapated_rand_error_resample_getsamples(nSamples, pCI, ares, precs, recs)

def adapted_rand_error_resample_objects(lbl_truth, lbl_proposed, nObjects, nSamples=100, pCI=0.05, getDistros=False):
    #import time

    ntObjs = lbl_truth.max(); assert( lbl_truth.shape == lbl_proposed.shape )

    #print('resample'); t = time.time()

    # pre-generate selects for objects
    # bernoulli resampling of the objects with mean rate set to get mean of nObjects
    selObj = (np.random.rand(ntObjs,nSamples) < (nObjects / float(ntObjs)))
    # generate confusion matrix once, then resample below by selecting rows (done for speed optimization)
    m_ij, n = adapted_rand_error_getconfusion( lbl_truth, lbl_proposed, nogtbg=True )

    ares = np.zeros((nSamples,),dtype=np.double)
    precs = np.zeros((nSamples,),dtype=np.double)
    recs = np.zeros((nSamples,),dtype=np.double)
    for s in range(nSamples):
        ares[s], precs[s], recs[s] = adapted_rand_error_confusion( m_ij[np.nonzero(selObj[:,s])[0],:] )

    #print('\tdone in %.4f s' % (time.time() - t))
    if getDistros:
        return ares, precs, recs
    else:
        return adapated_rand_error_resample_getsamples(nSamples, pCI, ares, precs, recs)

def adapated_rand_error_resample_getsamples(nSamples, pCI, ares, precs, recs):
    # get the middle sample and the specified confidence interval based on sorted are
    plo = int(np.floor(nSamples * pCI/2)-1); assert( plo > -1 ); phi = nSamples-plo-1; pmed = nSamples/2
    i = np.argsort(ares); ares = ares[i]; precs = precs[i]; recs = recs[i]
    #print(ares[pmed], precs[pmed], recs[pmed], ares[[plo,phi]], precs[[plo,phi]], recs[[plo,phi]])
    return ares[pmed], precs[pmed], recs[pmed], ares[[plo,phi]], precs[[plo,phi]], recs[[plo,phi]]

