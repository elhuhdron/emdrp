# For cube size 128 128 128
python driver-cpu-FRAG-superchunk.py --label_count 25000 --size_of_borders 2000 --size_of_edges 50000 --validate 1 --do_cpu_rag 1

## For cube size 256 256 256
python driver-cpu-FRAG-superchunk.py --label_count 25000 --size_of_borders 2000 --size_of_edges 500000 --validate 1 --do_cpu_rag 1


## For cube size 384 384 384
python driver-cpu-FRAG-superchunk.py --label_count 25000 --size_of_borders 2000 --size_of_edges 500000 --validate 1 --do_cpu_rag 1 

## For cube size 512 512 480 - does not fit in memory for python code - cpp extension works
python driver-cpu-FRAG-superchunk.py --label_count 5000 --size_of_borders 2000 --size_of_edges 1000000 

## For cube size 684 684 480 - 
python driver-cpu-FRAG-superchunk.py --label_count 5000 --size_of_borders 2000 --size_of_edges 4000000
