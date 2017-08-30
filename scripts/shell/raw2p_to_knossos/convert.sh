
# Script for converting 2p imaging raw data to knossos cubes (with mag1 - mag4)
# Alternative approach to the knossos-cuber that utilizes the emdrp toolset.
# NOTE: if data is in tiff format have to first manually convert tiff stacks to raw using fiji 

type='uint16'
cnt=0
# create output hdf5 size that is divisble by this
outpath='/run/media/watkinspv/My Passport/kara/PcdhXM72_for_Knossos-062017'
templatepath='/home/watkinspv/Downloads/kara_converted'
tmppath='/home/watkinspv/Downloads/kara_converted/tmp'

#declare -a paths=('061617' '061617' '061617' '061617' '061617' '062517')
#declare -a names=('Concatenated Stacks_004-red' 'Concatenated Stacks_005-red' '061617_006-red' 'Concatenated Stacks_007-red' 'Concatenated Stacks_008-red' 'Concatenated Stacks_002-red') 
#declare -a names_out=('004' '005' '006' '007' '008' '002')
#declare -a sizes=('2048 2048 255' '2048 2048 260' '2048 2048 119' '2048 2048 229' '2048 2048 300' '2048 2048 419')

#declare -a paths=('070717' '070717' '070717' '070717')
#declare -a names=('Concatenated Stacks_001-red' 'Concatenated Stacks_002-red' 'Concatenated Stacks_007-red' 'Concatenated Stacks_008-red')
#declare -a names_out=('001' '002' '007' '008')
#declare -a sizes=('2048 2048 367' '2048 2048 439' '2048 2048 471' '2048 2048 559')

declare -a paths=('081117' '081117' '081117' '082517' '082517' '082517' '082517' '082517' '082517' '082517')
declare -a names=('Concatenated Stacks_001-red' 'Concatenated Stacks_004-red' 'Concatenated Stacks_005-red' 'Concatenated Stacks_001-red' 'Concatenated Stacks_002-red' 'Concatenated Stacks_003-red' 'Concatenated Stacks_004-red' 'Concatenated Stacks_005-red' 'Concatenated Stacks_006-red' 'Concatenated Stacks_007-red')
declare -a names_out=('001' '004' '005' '001' '002' '003' '004' '005' '006' '007')
declare -a sizes=('2048 2048 480' '2048 2048 384' '2048 2048 349' '2048 2048 557' '2048 2048 554' '2048 2048 501' '2048 2048 501' '2048 2048 556' '2048 2048 520' '2048 2048 562')

iter=0
while [ $iter -lt ${#paths[@]} ]
do
    # create paths / copy over template knossos directory format
    coutpath=$outpath/${paths[$iter]}
    mkdir -p "$coutpath"
    coutname=${paths[$iter]}_${names_out[$iter]}
    coutpath=$outpath/${paths[$iter]}/$coutname
    rm -rf "$coutpath"
    cp -Rf $templatepath/knossos_template "$coutpath"

    size_dash=${sizes[$iter]// /-}
    size_comma=${sizes[$iter]// /,}
    # temporary hdf5 names
    tmph5=${tmppath}/${coutname}
    # get number of knossos chunks required for data (ceil)
    nchunks=$(python -c "import numpy as np; sz=np.array([$size_comma],dtype=np.int32); ck=np.array([128,128,128],dtype=np.int32); print('%d %d %d' % tuple((-(-sz // ck)).tolist()))")
    # round data size up to some larger knossos chunk multiple
    usize=$(python -c "import numpy as np; sz=np.array([${nchunks// /,}],dtype=np.int32); ck=np.array([128,128,128],dtype=np.int32); ck2=np.array([512,512,512],dtype=np.int32); print('%d %d %d' % tuple(((-(-sz*ck // ck2))*ck2).tolist()))")
    # get number of knossos chunks for size rounded up to larger chunk multiple
    nchunks=$(python -c "import numpy as np; sz=np.array([${usize// /,}],dtype=np.int32); ck=np.array([128,128,128],dtype=np.int32); print('%d %d %d' % tuple((-(-sz // ck)).tolist()))")

    # create mag1 hdf5
    rm -rf $tmph5.h5
    dpWriteh5.py --srcfile $tmph5.h5 --datasize ${usize} --inraw "${paths[$iter]}/${names[$iter]}-${size_dash}.raw" --scale 1 1 1 --chunksize 128 128 128 --dataset-out 2p --fillvalue 0 --dpW --size ${sizes[$iter]} --chunk 0 0 0 --data-type $type --inraw-bigendian --dpL

    # write out mag1
    dpCubeIter.py --volume_range_beg 0 0 0 --volume_range_end $nchunks --overlap 0 0 0 --cube_size 1 1 1 --cmd "dpLoadh5.py --srcfile $tmph5.h5 --dataset 2p --dtypeGray uint8 --dtypeGrayScale 2047 --dpL" --fileflags outraw --filepaths "'$coutpath/mag1'" --fileprefixes ${coutname}_mag1 --filepostfixes '.raw' --filepaths-affixes 1 > tmp_out_${cnt}.sh
    sh tmp_out_${cnt}.sh

    # create mag2 hdf5
    rm -rf ${tmph5}_mag2.h5
    dpResample.py --srcfile $tmph5.h5 --dataset 2p --dpRes --resample-dims 1 1 1 --factor 2 --downsample-op median --outfile ${tmph5}_mag2.h5 --volume_range_beg 0 0 0 --volume_range_end $nchunks --overlap 0 0 0 --cube_size 4 4 4

    # write out mag2
    nchunks2=$(python -c "import numpy as np; sz=np.array([${nchunks// /,}],dtype=np.int32); print('%d %d %d' % tuple((sz//2).tolist()))")    
    dpCubeIter.py --volume_range_beg 0 0 0 --volume_range_end $nchunks2 --overlap 0 0 0 --cube_size 1 1 1 --cmd "dpLoadh5.py --srcfile ${tmph5}_mag2.h5 --dataset 2p --dtypeGray uint8 --dtypeGrayScale 2047 --dpL" --fileflags outraw --filepaths "'$coutpath/mag2'" --fileprefixes ${coutname}_mag2 --filepostfixes '.raw' --filepaths-affixes 1 > tmp_out_${cnt}.sh
    sh tmp_out_${cnt}.sh

    # create mag4 hdf5
    rm -rf ${tmph5}_mag4.h5
    dpResample.py --srcfile $tmph5.h5 --dataset 2p --dpRes --resample-dims 1 1 1 --factor 4 --downsample-op median --outfile ${tmph5}_mag4.h5 --volume_range_beg 0 0 0 --volume_range_end $nchunks --overlap 0 0 0 --cube_size 4 4 4

    # write out mag4
    nchunks4=$(python -c "import numpy as np; sz=np.array([${nchunks// /,}],dtype=np.int32); print('%d %d %d' % tuple((sz//4).tolist()))")    
    dpCubeIter.py --volume_range_beg 0 0 0 --volume_range_end $nchunks4 --overlap 0 0 0 --cube_size 1 1 1 --cmd "dpLoadh5.py --srcfile ${tmph5}_mag4.h5 --dataset 2p --dtypeGray uint8 --dtypeGrayScale 2047 --dpL" --fileflags outraw --filepaths "'$coutpath/mag4'" --fileprefixes ${coutname}_mag4 --filepostfixes '.raw' --filepaths-affixes 1 > tmp_out_${cnt}.sh
    sh tmp_out_${cnt}.sh

    # create the knossos.conf file
    cusize=(${usize})
    echo -e "experiment name \"${coutname}_mag1\";\nboundary x ${cusize[0]};\nboundary y ${cusize[1]};\nboundary z ${cusize[2]};\nscale x 10.0;\nscale y 10.0;\nscale z 10.0;\nmagnification 1;" > "$coutpath/knossos.conf"

    # delete all the tmp hdf5 files
    rm -rf $tmph5.h5 ${tmph5}_mag2.h5 ${tmph5}_mag4.h5

    iter=`expr $iter + 1`
done

