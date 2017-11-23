
# run on blue in ~/gits/emdrp/recon/python

outpath=/home/watkinspv/Downloads/K0057_tracing/tmp
fnraw=/home/watkinspv/Downloads/K0057_D31_dsx3y3z1_cropped_to_gt.h5
inlabels=/home/watkinspv/Downloads/K0057_tracing/K0057_D31_dsx3y3z1_labels.h5
dataset=data_mag_x3y3z1
size='256 256 32'
declare -a chunks=("6 23 3" "16 19 15" "24 14 8" "13 18 15" "4 35 3" "10 11 18")
declare -a offsets=("0 0 0" "0 0 32" "0 0 0" "0 0 64" "96 96 64" "32 96 48")
declare -a outraws=("K0057_D31_dsx3y3z1_x6_y23_z3o0" "K0057_D31_dsx3y3z1_x16_y19_z15o32" "K0057_D31_dsx3y3z1_x24_y14_z8o0" "K0057_D31_dsx3y3z1_x13_y18_z15o64" "K0057_D31_dsx3y3z1_x4o96_y35o96_z3o64" "K0057_D31_dsx3y3z1_x10o32_y11o96_z18o48")

# only for validation
count=0
for chunk in "${chunks[@]}";
do
    echo processing $chunk
    dpLoadh5.py --srcfile $fnraw --chunk $chunk --size $size --offset ${offsets[$count]} --dataset $dataset --outraw $outpath/${outraws[$count]}.nrrd --dpL
    dpLoadh5.py --srcfile $inlabels --chunk $chunk --size $size --offset ${offsets[$count]} --dataset labels --outraw $outpath/${outraws[$count]}_labels.nrrd --dpL

    count=`expr $count + 1`
done
