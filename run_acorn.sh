export debugSearchFlag=0
#! /bin/bash



cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -B   build

#make -C build -j faiss
#make -C build utils
make -C build -j acorn


#../DiskANN/build/apps/utils/compute_filtered_groundtruth  --K 10 --base_file ./base.fbin --base_labels ./base_labels.txt --dist_fn l2 --data_type float --gt_file gt.bin --query_file query.fbin --query_label ./query_labels.txt 

##########################################
# TESTING SIFT1M and PAPER
##########################################
now=$(date +"%m-%d-%Y")


 # run  
N=1000000
nq=10 
gamma=12 
dataset=amazon
M=32 
M_beta=64
$gt = "gt_1m.bin.txt"
$efs = 0 # 0 for build, >1 for search only
parent_dir=${now}_${dataset}
mkdir ${parent_dir}
dir=${parent_dir}/MB${M_beta}
mkdir ${dir}

TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> ${dir}/summary_${dataset}_n=${N}_gamma=${gamma}.txt


./build/demos/acorn $N $nq ../$dataset/ . . $gt 100 $M_beta $efs  &>> ${dir}/summary_${dataset}_n=${N}_gamma=${gamma}.txt








