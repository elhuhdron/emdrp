
# run on blue in ~/gits/emdrp/recon/python

fnraw=/home/watkinspv/Downloads/K0057_D31_dsx3y3z1_cropped_to_labels.h5
inpath=/home/watkinspv/Downloads/K0057_tracing_combined/tmp
outlabels=/home/watkinspv/Downloads/K0057_D31_dsx3y3z1_labels.h5
dataset=data_mag_x3y3z1
rois=ROIs.txt

rm -rf $outlabels

declare -a sizes=('256 256 128' '256 256 128' '256 256 128' '128 256 128' '256 256 32' '256 256 32' '256 256 32')
declare -a chunks=("6 23 2" "16 19 15" "4 35 2" "4 11 14" "24 14 8" "13 18 15" "10 11 18")
declare -a offsets=("0 0 32" "0 0 32" "96 96 96"  "96 64 112" "0 0 0" "0 0 64" "32 96 48")

count=0
for chunk in "${chunks[@]}";
do

    # create the filename
    cchunk=($chunk); coffset=(${offsets[$count]})
    fn=`printf 'K0057_D31_dsx3y3z1_x%do%d_y%do%d_z%do%d' ${cchunk[0]} ${coffset[0]} ${cchunk[1]} ${coffset[1]} ${cchunk[2]} ${coffset[2]}`
    size=${sizes[$count]}

    echo processing $chunk
    dpWriteh5.py --srcfile $fnraw --chunk $chunk --size $size --offset ${offsets[$count]} --outfile $outlabels --dataset $dataset --dataset-out labels --inraw $inpath/${fn}_labels_clean.nrrd --data-type-out uint16 --fillvalue 65535 --inroi $rois --dpW

    count=`expr $count + 1`
done

