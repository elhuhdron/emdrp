
# run on blue in ~/gits/emdrp/recon/python

fnraw=/u/erjelli/link_scratch/ECS_tutorial/download/M0027_11_mag1_subvol_8x8x4chunks_at_x0012_y0014_z0002.h5
inpath=/u/erjelli/link_scratch/ECS_tutorial/download
outlabels=/u/erjelli/link_scratch/ECS_tutorial/download/M0027_11_labels.h5
dataset=data_mag1
rois=roi_M0027.txt

rm -rf $outlabels

declare -a sizes=('128 128 128' '128 128 128' '128 128 128' '128 128 128' '128 128 128' '128 128 128')
declare -a chunks=("13 15 3" "13 20 3" "16 17 4" "18 15 3" "18 20 3" "18 20 4")
declare -a offsets=("0 0 0" "0 0 0" "0 0 0" "0 0 0" "0 0 0" "0 0 0")

count=0
for chunk in "${chunks[@]}";
do

    # create the filename
    cchunk=($chunk); coffset=(${offsets[$count]})
    fn=`printf 'M0027_11_groundtruth_cleaned_x%04d_y%04d_z%04d' ${cchunk[0]} ${cchunk[1]} ${cchunk[2]}`
    size=${sizes[$count]}

    echo processing $chunk
    dpWriteh5.py --srcfile $fnraw --chunk $chunk --size $size --offset ${offsets[$count]} --outfile $outlabels --dataset $dataset --dataset-out labels --inraw $inpath/${fn}.tif --data-type-out uint16 --fillvalue 65535 --inroi $rois --dpW

    count=`expr $count + 1`
done

