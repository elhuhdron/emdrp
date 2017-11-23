
# run on blue in ~/gits/emdrp/recon/python

fnraw=/Data/datasets/raw/K0057_D31_dsx3y3z1.h5
dataset=data_mag_x3y3z1
fnpath=/home/watkinspv/Downloads/K0057_tracing_cubes/tmp
declare -a ctx_offset=(64 64 16)

declare -a sizes=('256 256 128' '256 256 128' '256 256 128' '128 256 128')
declare -a ctx_sizes=('384 384 160' '384 384 160' '384 384 160' '256 384 160')
declare -a chunks=("6 23 2" "16 19 15" "4 35 2" "4 11 14")
declare -a offsets=("0 0 32" "0 0 32" "96 96 96"  "96 64 112")

count=0
for chunk in "${chunks[@]}";
do
    echo processing $chunk

    # subtract context offset from offset
    coffset=(${offsets[$count]}); cnt=0
    while [ $cnt -lt 3 ]
    do
        coffset[$cnt]=`expr ${coffset[$cnt]} - ${ctx_offset[$cnt]}`
        cnt=`expr $cnt + 1`
    done
    # convert back to space delimited string
    ofst=$( IFS=' '; echo "${coffset[*]}" )

    # create the filename
    cchunk=($chunk); coffset=(${offsets[$count]})
    fn=`printf 'K0057_D31_dsx3y3z1_x%do%d_y%do%d_z%do%d' ${cchunk[0]} ${coffset[0]} ${cchunk[1]} ${coffset[1]} ${cchunk[2]} ${coffset[2]}`
    size=${sizes[$count]}
    ctx_size=${ctx_sizes[$count]}

    # load raw data and write out nrrd
    dpLoadh5.py --srcfile $fnraw --dataset $dataset --outraw $fnpath/${fn}.nrrd --chunk $chunk --size $ctx_size --offset ${ofst} --zerocorners 64 64 80
    
    count=`expr $count + 1`
done

