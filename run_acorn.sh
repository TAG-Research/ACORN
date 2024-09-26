export debugSearchFlag=0
#! /bin/bash



cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -B   build

#make -C build -j faiss
#make -C build utils
make -C build -j acorn


#../DiskANN/build/apps/utils/compute_filtered_groundtruth  --K 10 --base_file ./base.fbin --base_labels ./base_labels.txt --dist_fn l2 --data_type float --gt_file gt_1M.bin --query_file query.fbin --query_label ./query_labels.txt 

##########################################
# TESTING SIFT1M and PAPER
##########################################
now=$(date +"%m-%d-%Y")


 # run  
N=1000000
nq=5000 
gamma=12 # expansion factor
dataset=amazon
M_beta=16 # max degree, use same M and M_beta for now
gt="gt_1m.bin.txt"
efc=100 # build limit
efs=150 # search limit, 0 for build only, >1 for search only
parent_dir=${now}_${dataset}
mkdir ${parent_dir}
dir=${parent_dir}/MB${M_beta}
mkdir ${dir}

TZ='America/Los_Angeles' date +"Start time: %H:%M" &>> ${dir}/summary_${dataset}_n=${N}_nq=${nq}_efs=${efs}_gamma=${gamma}_M_beta=${M_beta}.txt


./build/demos/acorn $N $nq ../$dataset/ . . $gt $efc $M_beta $efs  &>> ${dir}/summary_${dataset}_n=${N}_nq=${nq}_efs=${efs}_gamma=${gamma}_M_beta=${M_beta}.txt








