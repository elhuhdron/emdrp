
# run on blue in ~/gits/emdrp/recon/python

fnraw=/Data/datasets/raw/k0725_dsx2y2z1.h5
outraw=/home/watkinspv/Downloads/k0725_dsx2y2z1_cropped_to_labels.h5
dataset=data_mag_x2y2z1
declare -a ctx_offset=(64 64 64)
rois=ROIs.txt

rm -rf $outraw

declare -a sizes=('128 128 128' '128 128 128')
declare -a chunks=("5 5 3" "5 6 3")
declare -a offsets=("0 64 0" "0 64 0")

count=0
for chunk in "${chunks[@]}";
do
    echo processing $chunk

    # subtract context offset from offset
    coffset=(${offsets[$count]}); csize=(${sizes[$count]}); cnt=0
    while [ $cnt -lt 3 ]
    do
        coffset[$cnt]=`expr ${coffset[$cnt]} - ${ctx_offset[$cnt]}`
        csize[$cnt]=`expr ${csize[$cnt]} + 2 \* ${ctx_offset[$cnt]}`
        cnt=`expr $cnt + 1`
    done
    # convert back to space delimited strings
    ofst=$( IFS=' '; echo "${coffset[*]}" ); sz=$( IFS=' '; echo "${csize[*]}" )

    dpWriteh5.py --srcfile $fnraw --chunk $chunk --size $sz --offset $ofst --outfile $outraw --dataset $dataset --inroi $rois --dpW

    count=`expr $count + 1`
done
