
# run on blue in ~/gits/emdrp/recon/python

outpath=/home/watkinspv/Downloads/tmp_validate
fnraw=/home/watkinspv/Downloads/k0725_dsx2y2z1_cropped_to_labels.h5
inlabels=/Data/datasets/labels/gt/k0725_labels_dsx2y2z1.h5
dataset=data_mag_x2y2z1

declare -a sizes=('128 128 128' '128 128 128')
declare -a chunks=("5 5 3" "5 6 3")
declare -a offsets=("0 64 0" "0 64 0")

# only for validation
count=0
for chunk in "${chunks[@]}";
do
    # create the filename
    cchunk=($chunk); coffset=(${offsets[$count]})
    fn=`printf 'k0725_dsx2y2z1_x%do%d_y%do%d_z%do%d' ${cchunk[0]} ${coffset[0]} ${cchunk[1]} ${coffset[1]} ${cchunk[2]} ${coffset[2]}`
    size=${sizes[$count]}

    echo processing $chunk
    dpLoadh5.py --srcfile $fnraw --chunk $chunk --size $size --offset ${offsets[$count]} --dataset $dataset --outraw $outpath/${fn}.nrrd --dpL
    dpLoadh5.py --srcfile $inlabels --chunk $chunk --size $size --offset ${offsets[$count]} --dataset labels --outraw $outpath/${fn}_labels.nrrd --dpL

    count=`expr $count + 1`
done
