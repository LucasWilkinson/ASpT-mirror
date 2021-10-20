#!/bin/bash

cd SpMM_KNL
make
cd ..

cd SDDMM_KNL
make
cd ..

cd CSB
make spmm_sall
make spmm_dall
cd ..

#iSpMM_ASpT_SP
#SpMM_MKL_SP
#SpMM_MKL_DP


