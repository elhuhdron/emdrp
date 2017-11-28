
# run on blue in ~/gits/emdrp/recon/python

fnraw=/home/watkinspv/Data/ECS_tutorial/M0007_33_8x8x4chunks_at_x0016_y0017_z0000.h5
inpath=/home/watkinspv/Data/ECS_tutorial
outlabels=/home/watkinspv/Data/ECS_tutorial/M0007_33_labels.h5
dataset=data_mag1
rois=roi_M0007.txt

rm -rf $outlabels

declare -a sizes=('128 128 128' '128 128 128' '128 128 128' '128 128 128' '128 128 128' '128 128 128')
declare -a chunks=("17 19 2" "17 23 1" "19 22 2" "22 18 1" "22 23 1" "22 23 2")
declare -a offsets=("0 0 0" "0 0 0" "0 0 0" "0 0 0" "0 0 0" "0 0 0")

count=0
for chunk in "${chunks[@]}";
do

    # create the filename
    cchunk=($chunk); coffset=(${offsets[$count]})
    #fn=`printf 'M0007_33_groundtruth_cleaned_x%do%d_y%do%d_z%do%d' ${cchunk[0]} ${coffset[0]} ${cchunk[1]} ${coffset[1]} ${cchunk[2]} ${coffset[2]}`
    fn=`printf 'M0007_33_groundtruth_cleaned_x%04d_y%04d_z%04d' ${cchunk[0]} ${cchunk[1]} ${cchunk[2]}`
    size=${sizes[$count]}

    echo processing $chunk
    dpWriteh5.py --srcfile $fnraw --chunk $chunk --size $size --offset ${offsets[$count]} --outfile $outlabels --dataset $dataset --dataset-out labels --inraw $inpath/${fn}.nrrd --data-type-out uint16 --fillvalue 65535 --inroi $rois --dpW

    count=`expr $count + 1`
done

