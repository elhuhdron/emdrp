
# to export raw nrrd (using emdrp toolset):
#dpLoadh5.py --srcfile /mnt/cne/from_externals/K0057_D31/K0057_D31.h5 --chunk 90 78 15 --size 192 192 32 --offset 96 96 96 --dataset data_mag1 --outraw ~/Downloads/K0057_webknossos_test/K0057_D31_x90o96_y78o96_z15o96.nrrd --dpL

# script to read the webknossos formatted label volume and export as nrrd.

# pip install wkw
import wkw
# pip install pynrrd
import nrrd
import numpy as np
import os

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

def make_consensus( votes, cnt, tie_include=False ):
    if (cnt % 2) or not tie_include:
        # odd number of users in consensus, no ties, must have majority
        # OR even number of users in consensus, tie excludes voxel from consensus
        return (votes > cnt/2)
    else:
        # even number of users in consensus, tie includes voxel in consensus
        return (votes >= cnt/2)

# top level path to data files
dn = '/home/pwatkins/Downloads/K0057_webknossos_test'
#dn = '.'

## location of the nrrd raw data file
raw_fn = os.path.join(dn, 'K0057_D31_x90o96_y78o96_z15o96.nrrd')

# relative paths of the stored wkw labels (top dir).
# NOTE: for the normal workflow of each labeler only tracing one neurite per volume,
#   this needs to be two dimensional (i.e. indexed by both the label and labeler)
wkw_labels = ['data/1', 'data/1']

# filename where to store the nrrd formatted label data
out_labels = os.path.join(dn, 'K0057_D31_x90o96_y78o96_z15o96_labels.nrrd')

# location of the labels within the dataset
labeled_coordinate = [11616, 10080, 2016]
labeled_size = [192, 192, 32]

# which label values to make consensuses for for each labeled volume.
# a workflow would typically only have one object labeled per user labeled volume.
consensus_label_values = [2,3,4]

nlabels = len(consensus_label_values)
nlabelers = len(wkw_labels)
fScores = np.zeros((nlabels,nlabelers), dtype=np.double)
consensus_labels = np.zeros(labeled_size, dtype=np.int32)
for j in range(nlabels):

    # code to make a consensus for a single labeled neurite or object.
    votes = np.zeros(labeled_size, dtype=np.int64)
    for i in range(nlabelers):
        fn = os.path.join(dn, wkw_labels[i])
        dataset = wkw.Dataset.open(fn)
        # xxx - what is the first singleton dimension?
        data = np.squeeze(dataset.read(labeled_coordinate, labeled_size))

        # binarized_data represents the "votes" for the current labeler (i) for the current label (j)
        binarized_data = (data == consensus_label_values[j]) # if all the user labels values are the same
        #binarized_data = (data > 0) # if only one label is stored per labeled volume

        # running sum of the "votes" per voxel
        votes = votes + binarized_data

    consensus = make_consensus(votes, nlabelers)
    # add the consensus for this label to the output label volume
    # NOTE: this does not handle the situation if a voxel is in more than one consensus.
    #   this should not typically happen, so just overwrite in this case.
    sel = (consensus > 0)
    consensus_labels[sel] = (consensus.astype(np.uint32) * consensus_label_values[j])[sel]

    # calculate a score for each user
    for i in range(nlabelers):
        fn = os.path.join(dn, wkw_labels[i])
        dataset = wkw.Dataset.open(fn)
        # xxx - what is the first singleton dimension?
        data = np.squeeze(dataset.read(labeled_coordinate, labeled_size))

        # if each label is stored sepearately
        binarized_data = (data == consensus_label_values[j])

        fScores[j,i], tpr_recall, precision, tp, fp, fn = pix_fscore_metric( consensus, binarized_data )

print('User fScores:')
print(fScores)

# read in the raw data format to keep the same nrrd header
rawdata, header = nrrd.read(raw_fn)
#print(rawdata.shape, rawdata.dtype, data.shape, data.dtype)
#print(header)

# write out the consensus labels
header['type'] = 'uint32'
nrrd.write(out_labels, consensus_labels, header)
