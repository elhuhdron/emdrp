
fnraw=/Data/datasets/raw/K0057_D31_dsx3y3z1.h5
inpath=/home/watkinspv/Downloads/K0057_tracing
outlabels=/home/watkinspv/Downloads/K0057_tracing/K0057_D31_dsx3y3z1_labels.h5
dataset=data_mag_x3y3z1
size='256 256 32'
rois=tmp_roi_k0057_dsx3y3z1.txt
declare -a chunks=("6 23 3" "16 19 15" "24 14 8" "13 18 15")
declare -a offsets=("0 0 0" "0 0 32" "0 0 0" "0 0 64")
declare -a inraws=("K0057_D31_dsx3y3z1_x6_y23_z3o0_labels_clean.nrrd" "K0057_D31_dsx3y3z1_x16_y19_z15o32_labels_clean.nrrd" "K0057_D31_dsx3y3z1_x24_y14_z8o0_labels_clean.nrrd" "K0057_D31_dsx3y3z1_x13_y18_z15o64_labels_clean.nrrd")

count=0
for chunk in "${chunks[@]}";
do
    echo processing $chunk
    dpWriteh5.py --srcfile $fnraw --chunk $chunk --size $size --offset ${offsets[$count]} --outfile $outlabels --dataset $dataset --dataset-out labels --inraw $inpath/${inraws[$count]} --data-type-out uint16 --fillvalue 65535 --inroi $rois --dpW

    count=`expr $count + 1`
done
