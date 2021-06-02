
# change these to proper locations
inpath=/mnt/cne/pwatkins/from_externals/ECS_paper
outpath=$HOME/Downloads/Bahar_GAN_tmp

# do not change, original raw hdf5 files
fnraws=( M0027_11_33x37x7chunks_Forder.h5 M0007_33_39x35x7chunks_Forder.h5 )

# call these what you will (inputs), see comment below "export the existing..."
fnvolumeinputs=( M0027_my_new_raw_data_hotness.nrrd M0007_my_new_raw_data_hotness.nrrd )
# call these what you will (outputs)
fnouts=( M0027_my_new_raw_data_hotness.h5 M0007_my_new_raw_data_hotness.h5 )
# call these what you will (outputs), this is a sanity check that rewrites out volume files.
# recommend loading these back in fiji to make sure it looks like you expect.
fnsanityouts=( M0027_my_new_raw_data_hotness_sanity.nrrd M0007_my_new_raw_data_hotness_sanity.nrrd )

# do not change
dataset=data_mag1

# these need to change depending on context.
# arrays are paralle to fnraws/fnouts array above.
# for NO context (original locations)
#sizes=( '1024 1024 512' '1024 1024 512' )
#chunks=( '12 14 2' '16 17 0' )
#offsets=( '0 0 0' '0 0 0' )
# for 128 context on all possible sides
sizes=( '1280 1280 768' '1280 1280 640' )
chunks=( '11 13 1' '15 16 0' )
offsets=( '0 0 0' '0 0 0' )

for i in ${!fnraws[@]}; do
    # export the existing data to an F-order raw file.
    # this needs to be replaced with generation of the NEW volume.
    # this is simply a placeholder for creating a new raw volume to write below.
    # easiest method would probably be to import tiff stack into fiji and then saveas nrrd...
    dpLoadh5.py --srcfile $inpath/${fnraws[$i]} --chunk ${chunks[$i]} --size ${sizes[$i]} --offset ${offsets[$count]} --dataset $dataset --outraw $outpath/${fnvolumeinputs[$i]} --dpL

    # write the new volume to a new hdf files. original file is only a template (copies over attributes and size).
    rm -rf $outpath/${fnouts[$i]} # dpWrite does not overwrite h5, can cause problems sometimes so just remove
    dpWriteh5.py --srcfile $inpath/${fnraws[$i]} --outfile $outpath/${fnouts[$i]} --chunk ${chunks[$i]} --size ${sizes[$i]} --offset ${offsets[$count]} --dataset $dataset --inraw $outpath/${fnvolumeinputs[$i]} --dpW

    # write out a new volume file as a sanity check that it matches the input (check with fiji)
    # comment if you are sufficiently sane.
    dpLoadh5.py --srcfile $outpath/${fnouts[$i]} --chunk ${chunks[$i]} --size ${sizes[$i]} --offset ${offsets[$count]} --dataset $dataset --outraw $outpath/${fnsanityouts[$i]} --dpL
done

