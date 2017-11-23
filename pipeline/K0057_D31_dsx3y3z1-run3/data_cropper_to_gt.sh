
# run on blue in ~/gits/emdrp/recon/python

fnraw=/Data/datasets/raw/K0057_D31_dsx3y3z1.h5
outraw=/home/watkinspv/Downloads/K0057_D31_dsx3y3z1_cropped_to_gt.h5
dataset=data_mag_x3y3z1
size='320 320 96'
ctx_offset=32
rois=ROIs.txt
declare -a chunks=("6 23 3" "16 19 15" "24 14 8" "13 18 15" "4 35 3" "10 11 18")
declare -a offsets=("0 0 0" "0 0 32" "0 0 0" "0 0 64" "96 96 64" "32 96 48")

count=0
for chunk in "${chunks[@]}";
do
    echo processing $chunk

    # subtract context offset from offset
    coffset=(${offsets[$count]}); cnt=0
    while [ $cnt -lt 3 ]
    do
        coffset[$cnt]=`expr ${coffset[$cnt]} - $ctx_offset`
        cnt=`expr $cnt + 1`
    done
    # convert back to space delimited string
    ofst=$( IFS=' '; echo "${coffset[*]}" )

    dpWriteh5.py --srcfile $fnraw --chunk $chunk --size $size --offset $ofst --outfile $outraw --dataset $dataset --inroi $rois --dpW

    count=`expr $count + 1`
done
