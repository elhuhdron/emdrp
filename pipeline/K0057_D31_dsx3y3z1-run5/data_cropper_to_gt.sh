
# run on blue in ~/gits/emdrp/recon/python

fnraw=/Data/datasets/raw/K0057_D31_dsx3y3z1.h5
outraw=/home/watkinspv/Downloads/K0057_D31_dsx3y3z1_cropped_to_labels.h5
dataset=data_mag_x3y3z1
declare -a ctx_offset=(64 64 16)
rois=ROIs.txt

rm -rf $outraw

declare -a sizes=('256 256 128' '256 256 128' '256 256 128' '128 256 128' '256 256 32' '256 256 32' '256 256 32')
declare -a chunks=("6 23 2" "16 19 15" "4 35 2" "4 11 14" "24 14 8" "13 18 15" "10 11 18")
declare -a offsets=("0 0 32" "0 0 32" "96 96 96"  "96 64 112" "0 0 0" "0 0 64" "32 96 48")

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
