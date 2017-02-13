
fn=/home/watkinspv/Downloads/k0725_supervoxels_x0013_y0033_z0033o32.nrrd
fnout=/home/watkinspv/Downloads/k0725_proof_x0013_y0033_z0033o32.nrrd
fnovl=/home/watkinspv/Downloads/k0725_overlay_x0013_y0033_z0033o32.nrrd
size='384 384 96'
corners='192 192 32'

# NOTES: nrrd has to be created with label_cropper.sh so that ECS supervoxels are reassigned.
#   Without ECS, can just use outraw from dpLoadh5.
#   Currently this script is setup for without ECS.

# for the first run, relabel starting with supervoxels
#dpCleanLabels.py --inraw $fn --size $size --relabel --min-label 2 --fg-connectivity 3 --srcfile '' --outraw $fnout \
#    --zerocorners $corners --ECS-label 0 --dpC

# for subsequent runs, do path calculation then relabel
dpCleanLabels.py --inraw $fnout --size $size --fg-connectivity 3 --srcfile '' --outraw $fnovl --minpath 1 \
    --ECS-label 0 --minpath-skel --dpC
dpCleanLabels.py --inraw $fnout --size $size --relabel --min-label 2 --fg-connectivity 3 --srcfile '' --outraw $fnout \
    --zerocorners $corners --ECS-label 0 --dpC

# for both, always remove small labels
dpCleanLabels.py --inraw $fnout --size $size --srcfile '' --outraw $fnout --minsize 28 --min-label 2 --ECS-label 0 --dpC
