
fnout=/home/watkinspv/Downloads/K0057_bootstrapping/K0057_D31_dsx3y3z1_x4o96_y35o96_z3_proof.nrrd
fnovl=/home/watkinspv/Downloads/K0057_bootstrapping/K0057_D31_dsx3y3z1_x4o96_y35o96_z3_overlay.nrrd
size='384 384 160'
corners='64 64 32'

# proofer steps with ECS, create input using label_cropper.sh first
# NOTE, set minlabel==3 when exporting

# for subsequent runs, do path calculation then relabel
dpCleanLabels.py --inraw $fnout --size $size --fg-connectivity 3 --srcfile '' --outraw $fnovl --minpath 2 \
    --ECS-label 1 --minpath-skel --dpC
dpCleanLabels.py --inraw $fnout --size $size --relabel --min-label 2 --fg-connectivity 3 --srcfile '' --outraw $fnout \
    --zerocorners $corners --ECS-label 1 --dpC

# for both, always remove small labels
dpCleanLabels.py --inraw $fnout --size $size --srcfile '' --outraw $fnout --minsize 9 --min-label 2 --ECS-label 1 --dpC
