#for cube size 128 128 128
python driver-cpu-FRAG-superchunk.py --label_count 25000 --size_of_borders 2000 --size_of_edges 50000 --validate 1 --do_cpu_rag 1 --batch_edges 50000 --blockdim 8 --label_jump_borders 500 --batch_borders 11000


# for cube size 256 256 256
python driver-cpu-FRAG-superchunk.py --label_count 25000 --size_of_borders 2000 --size_of_edges 500000 --batch_edges 400000 --blockdim 8 --batch_borders 14000 --do_cpu_rag 1 --validate 1 --label_jump_borders 2000

#for cube size 384 384 384
python driver-cpu-FRAG-superchunk.py --label_count 25000 --size_of_borders 2000 --size_of_edges 500000 --batch_edges 400000 --blockdim 8 --batch_borders 14000 --do_cpu_rag 1 --validate 1 --label_jump_borders 4000

#for cube size 512 512 480

python driver-cpu-FRAG-superchunk.py --label_count 25000 --size_of_borders 2000 --size_of_edges 1000000 --batch_edges 400000 --blockdim 8 --batch_borders 14000 --label_jump_borders 7000 


#for cube 640 640 480 

python driver-cpu-FRAG-superchunk.py --label_count 25000 --size_of_borders 2000 --size_of_edges 2000000 --batch_edges 400000 --blockdim 8 --batch_borders 17000 --label_jump_borders 7000


#Not explored for pinned
#cube 1024 1024 1024
python driver-cpu-FRAG-superchunk.py --label_count 25000 --size_of_borders 500 --size_of_edges 4000000 --batch_edges 400000 --blockdim 8 --batch_borders 17000 --label_jump_borders 3000

