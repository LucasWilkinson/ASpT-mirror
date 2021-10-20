#!/bin/bash

cd cuSPARSE_SpMM
nvcc -O3 -gencode arch=compute_60,code=sm_60 -lcublas -lcusparse cuSPARSE_SP.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o cuSPARSE_SP
nvcc -O3 -gencode arch=compute_60,code=sm_60 -lcublas -lcusparse cuSPARSE_DP.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o cuSPARSE_DP
cd ..

