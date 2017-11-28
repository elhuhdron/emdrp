
# run on blue in ~/gits/emdrp/recon/python

outpath=/home/watkinspv/Data/ECS_tutorial/validate
fnraw=/home/watkinspv/Data/ECS_tutorial/M0027_11_8x8x4chunks_at_x0012_y0014_z0002.h5
inlabels=/home/watkinspv/Data/ECS_tutorial/M0027_11_labels.h5
dataset=data_mag1

declare -a sizes=('128 128 128' '128 128 128' '128 128 128' '128 128 128' '128 128 128' '128 128 128')
declare -a chunks=("13 15 3" "13 20 3" "16 17 4" "18 15 3" "18 20 3" "18 20 4")
declare -a offsets=("0 0 0" "0 0 0" "0 0 0" "0 0 0" "0 0 0" "0 0 0")

# only for validation
count=0
for chunk in "${chunks[@]}";
do
    # create the filename
    cchunk=($chunk); coffset=(${offsets[$count]})
    fn=`printf 'M0027_dsx3y3z1_x%do%d_y%do%d_z%do%d' ${cchunk[0]} ${coffset[0]} ${cchunk[1]} ${coffset[1]} ${cchunk[2]} ${coffset[2]}`
    size=${sizes[$count]}

    echo processing $chunk
    dpLoadh5.py --srcfile $fnraw --chunk $chunk --size $size --offset ${offsets[$count]} --dataset $dataset --outraw $outpath/${fn}.nrrd --dpL
    dpLoadh5.py --srcfile $inlabels --chunk $chunk --size $size --offset ${offsets[$count]} --dataset labels --outraw $outpath/${fn}_labels.nrrd --dpL

    count=`expr $count + 1`
done
