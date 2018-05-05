
chunkz=0
while [ $chunkz -lt 79 ]
do
    # for test
    #dpLoadh5.py --srcfile /mnt/cne/from_externals/110629_k0725/k0725.h5 --chunk 10 10 $chunkz --size 1024 1024 128 --dataset data_mag1 --outraw /mnt/ext/110629_k0725/pngs.png --dpL
    
    #dpLoadh5.py --srcfile /mnt/cne/from_externals/110629_k0725/k0725.h5 --chunk 0 0 $chunkz --size 4992 16000 128 --dataset data_mag1 --outraw /mnt/ext/110629_k0725/tiffs.tiff --dpL
    dpLoadh5.py --srcfile /mnt/cne/from_externals/110629_k0725/k0725.h5 --chunk 0 0 $chunkz --size 4992 16000 128 --dataset data_mag1 --outraw /mnt/ext/110629_k0725/pngs.png --dpL

    chunkz=`expr $chunkz + 1`
done

