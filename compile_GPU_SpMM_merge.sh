#!/bin/bash

rm -Rf merge-spmm
git clone --recursive https://github.com/owensgroup/merge-spmm.git
cp gbspmm.cu ./merge-spmm/test

cd merge-spmm
cd ext/merge-spmv
make gpu_spmv sm=600
cd ../../
cmake .
make
cd ..

