
# run matlab code volume_diameter_fit.m to export soma_cuts.mat

# export new gipl with cuts applied
python soma_axes_anal.py 

# create h5 from cut gipl
dpWriteh5.py --srcfile ~/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4-clean.h5 --outfile ~/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4-clean-cut.h5 --inraw ~/Downloads/K0057_soma_annotation/out/K0057_D31_dsx12y12z4_somas_clean_cut.gipl --dpL --dpW --chunk 0 0 0 --size 1696 1440 640 --dataset labels

# mesh cut h5
python -u $HOME/gits/emdrp/recon/python/dpLabelMesher.py --dataset labels --dpLabelMesher-verbose --set-voxel-scale --dataset-root 0 --reduce-frac 0.1 --smooth 5 5 5 --contour-lvl 0.25 --mesh-outfile /home/watkinspv/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4-clean-cut.0.mesh.h5 --srcfile ~/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4-clean-cut.h5 --chunk 0 0 0 --size 1696 1440 640

# create mat for soma_anal.m
python soma_celltype_export.py 

# view cut meshes with vtk
dpLabelMesher.py --mesh-infiles ~/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4-clean-cut.0.mesh.h5 --show-all --srcfile ~/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4-clean-cut.h5 --dataset-root 0 --lut-file ~/Downloads/distinguishable_bright.lut --opacity 0.5

# move all regular files to subdir, then all *cut* to regular, then iterate

