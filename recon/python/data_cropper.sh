
fnraw=/Data/datasets/raw/K0057_D31_dsx3y3z1.h5
outraw=/home/watkinspv/Downloads/K0057_D31_dsx3y3z1_cropped_to_labels.h5
dataset=data_mag_x3y3z1
size='512 512 384'
rois=tmp_roi_k0057_dsx3y3z1.txt
declare -a chunks=("5 22 2" "15 18 14" "23 13 7")

count=0
for chunk in "${chunks[@]}";
do
    echo processing $chunk
    dpWriteh5.py --srcfile $fnraw --chunk $chunk --size $size --outfile $outraw --dataset $dataset --inroi $rois --dpW

    count=`expr $count + 1`
done
