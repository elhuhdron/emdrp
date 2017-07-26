
# run on red or machine with mounted synology

outpath=/home/watkinspv/Downloads/tmp_probs
inprobs=/mnt/syn2/watkinspv/full_datasets/neon_xfold/mfergus32_K0057_ds3_run4/K0057-dsx3y3z1_xyz
dataset=data_mag_x3y3z1
ctx_offset=32

declare -a sizes=('256 256 128' '256 256 128' '256 256 128' '128 256 128')
declare -a chunks=("6 23 2" "16 19 15" "4 35 2" "4 11 14")
declare -a offsets=("0 0 32" "0 0 32" "96 96 96"  "96 64 112")

# only for validation
count=0
for chunk in "${chunks[@]}";
do
    # create the filename
    cchunk=($chunk); coffset=(${offsets[$count]})
    fn=`printf 'K0057_D31_dsx3y3z1_x%do%d_y%do%d_z%do%d' ${cchunk[0]} ${coffset[0]} ${cchunk[1]} ${coffset[1]} ${cchunk[2]} ${coffset[2]}`

    # subtract context offset from offset
    coffset=(${offsets[$count]}); csize=(${sizes[$count]}); cnt=0
    while [ $cnt -lt 3 ]
    do
        coffset[$cnt]=`expr ${coffset[$cnt]} - $ctx_offset`
        csize[$cnt]=`expr ${csize[$cnt]} + 2 \* $ctx_offset`
        cnt=`expr $cnt + 1`
    done
    # convert back to space delimited strings
    ofst=$( IFS=' '; echo "${coffset[*]}" ); sz=$( IFS=' '; echo "${csize[*]}" )

    cnt=0
    while [ $cnt -lt 4 ]
    do
        echo processing $chunk count $cnt
        dpLoadh5.py --srcfile ${inprobs}_$cnt.h5 --chunk $chunk --size $sz --offset $ofst --dataset MEM --outraw $outpath/${fn}_MEM_$cnt.nrrd --dpL
        #dpLoadh5.py --srcfile ${inprobs}_$cnt.h5 --chunk $chunk --size $sz --offset $ofst --dataset ICS --outraw $outpath/${fn}_ICS_$cnt.nrrd --dpL
        #dpLoadh5.py --srcfile ${inprobs}_$cnt.h5 --chunk $chunk --size $sz --offset $ofst --dataset ECS --outraw $outpath/${fn}_ECS_$cnt.nrrd --dpL
        cnt=`expr $cnt + 1`
    done

    count=`expr $count + 1`
done
