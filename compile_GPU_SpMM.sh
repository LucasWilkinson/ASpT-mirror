#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(readlink -f $(dirname /uufs/chpc.utah.edu/sys/installdir/cuda/9.1.85/bin/nvcc)/../lib64)
export CUDA_HOME=$(readlink -f $(dirname /uufs/chpc.utah.edu/sys/installdir/cuda/9.1.85/bin/nvcc)/..)
echo $LD_LIBRARY_PATH
echo $CUDA_HOME

mkdir build; cd build;
wget -nc http://sourceforge.net/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.gz
tar xf boost_1_58_0.tar.gz
cd boost_1_58_0
./bootstrap.sh --prefix=`readlink -f ..`
./b2 --with-program_options
cd .. # // boost_1_58_0
cd .. # // build

echo `pwd`

git submodule update --init --recursive
cp gbspmm.cu ./merge-spmm/test

cd merge-spmm
cd ext/merge-spmv
make gpu_spmv sm=600
cd ../../
cmake -DBOOST_INCLUDEDIR=`readlink -f ../build/boost_1_58_0` -DBOOST_LIBRARYDIR=`readlink -f ../build/boost_1_58_0/stage/lib` .
make
cd ..

cd ASpT_SpMM_GPU
nvcc -std=c++11 -O3 -gencode arch=compute_60,code=sm_60 sspmm_32.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o sspmm_32
nvcc -std=c++11 -O3 -gencode arch=compute_60,code=sm_60 sspmm_128.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o sspmm_128
nvcc -std=c++11 -O3 -gencode arch=compute_60,code=sm_60 dspmm_32.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o dspmm_32
nvcc -std=c++11 -O3 -gencode arch=compute_60,code=sm_60 dspmm_128.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o dspmm_128
cd ..

cd cuSPARSE_SpMM
nvcc -O3 -gencode arch=compute_60,code=sm_60 -lcublas -lcusparse cuSPARSE_SP.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o cuSPARSE_SP
nvcc -O3 -gencode arch=compute_60,code=sm_60 -lcublas -lcusparse cuSPARSE_DP.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o cuSPARSE_DP
cd ..

