
rm ~/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4-clean-cut-fit-ellipses.h5
rm ~/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4-clean-cut-fit-ellipses.0.mesh.h5

python soma_axes_anal.py 

dpWriteh5.py --srcfile ~/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4-clean-cut.h5 --outfile ~/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4-clean-cut-fit-ellipses.h5 --inraw ~/Downloads/K0057_soma_annotation/out/K0057_D31_dsx12y12z4_somas_clean_cut_fit_ellipses.gipl --dpL --dpW --chunk 0 0 0 --size 1696 1440 640 --dataset labels

python -u $HOME/gits/emdrp/recon/python/dpLabelMesher.py --dataset labels --dpLabelMesher-verbose --set-voxel-scale --dataset-root 0 --reduce-frac 0.1 --smooth 5 5 5 --contour-lvl 0.25 --mesh-outfile /home/watkinspv/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4-clean-cut-fit-ellipses.0.mesh.h5 --srcfile ~/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4-clean-cut-fit-ellipses.h5 --chunk 0 0 0 --size 1696 1440 640

python soma_celltype_export.py

