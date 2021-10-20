#!/bin/bash
git submodule update --init
cp Learner.scala ./BIDMach/src/main/scala/BIDMach
cd BIDMach
mvn package
cd ..

cd ASpT_SDDMM_GPU
nvcc -std=c++11 -O3 -gencode arch=compute_60,code=sm_60 sddmm_32.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o sddmm_32
nvcc -std=c++11 -O3 -gencode arch=compute_60,code=sm_60 sddmm_128.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o sddmm_128
cd .. 

cd BIDMach
mvn package 
cd ..
