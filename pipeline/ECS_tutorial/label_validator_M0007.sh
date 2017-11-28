
# run on blue in ~/gits/emdrp/recon/python

outpath=/home/watkinspv/Data/ECS_tutorial/validate
fnraw=/home/watkinspv/Data/ECS_tutorial/M0007_33_8x8x4chunks_at_x0016_y0017_z0000.h5
inlabels=/home/watkinspv/Data/ECS_tutorial/M0007_33_labels.h5
dataset=data_mag1

declare -a sizes=('128 128 128' '128 128 128' '128 128 128' '128 128 128' '128 128 128' '128 128 128')
declare -a chunks=("17 19 2" "17 23 1" "19 22 2" "22 18 1" "22 23 1" "22 23 2")
declare -a offsets=("0 0 0" "0 0 0" "0 0 0" "0 0 0" "0 0 0" "0 0 0")

# only for validation
count=0
for chunk in "${chunks[@]}";
do
    # create the filename
    cchunk=($chunk); coffset=(${offsets[$count]})
    fn=`printf 'M0007_dsx3y3z1_x%do%d_y%do%d_z%do%d' ${cchunk[0]} ${coffset[0]} ${cchunk[1]} ${coffset[1]} ${cchunk[2]} ${coffset[2]}`
    size=${sizes[$count]}

    echo processing $chunk
    dpLoadh5.py --srcfile $fnraw --chunk $chunk --size $size --offset ${offsets[$count]} --dataset $dataset --outraw $outpath/${fn}.nrrd --dpL
    dpLoadh5.py --srcfile $inlabels --chunk $chunk --size $size --offset ${offsets[$count]} --dataset labels --outraw $outpath/${fn}_labels.nrrd --dpL

    count=`expr $count + 1`
done
