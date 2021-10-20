#!/bin/bash

cd ASpT_SpMM_GPU
nvcc -std=c++11 -O3 -gencode arch=compute_60,code=sm_60 sspmm_32.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o sspmm_32
nvcc -std=c++11 -O3 -gencode arch=compute_60,code=sm_60 sspmm_128.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o sspmm_128
nvcc -std=c++11 -O3 -gencode arch=compute_60,code=sm_60 dspmm_32.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o dspmm_32
nvcc -std=c++11 -O3 -gencode arch=compute_60,code=sm_60 dspmm_128.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o dspmm_128
cd ..

