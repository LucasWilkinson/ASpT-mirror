#!/bin/bash
source /home/kunal/intel/compilers_and_libraries_2018/linux/bin/compilervars.sh intel64

mkdir KNL_result

cd data
rm SpMM_KNL_SP.out
rm SpMM_KNL_DP.out
rm SpMM_KNL_SP_preprocessing.out
rm SpMM_KNL_DP_preprocessing.out
rm SDDMM_KNL_SP.out
rm SDDMM_KNL_DP.out


echo "dataset, MKL_GFLOPs(K=32), MKL_GFLOPs(K=128), TACO_GFLOPs(K=32), TACO_GFLOPs(K=128), CSB_GFLOPs(K=32), CSB_GFLOPs(K=128), ASpT_GFLOPs(K=32), ASpT_GFLOPs(K=128), ASpT_diff_%(K=32+K=128)" >> SpMM_KNL_SP.out
echo "dataset, preprocessing_ratio" >> SpMM_KNL_SP_preprocessing.out

for i in `ls -d */`
do
cd ${i}
ii=${i/\//}
cd ..
echo -n ${ii} >> SpMM_KNL_SP.out
echo -n "," >> SpMM_KNL_SP.out
echo -n ${ii} >> SpMM_KNL_SP_preprocessing.out
echo -n "," >> SpMM_KNL_SP_preprocessing.out
../SpMM_KNL/SpMM_MKL_SP.x --input=./${ii}/${ii}.mtx
../SpMM_KNL/SpMM_TACO_SP.x --input=./${ii}/${ii}.mtx
../CSB/spmm_s32 ${ii}/${ii}.mtx
../CSB/spmm_s128 ${ii}/${ii}.mtx
../SpMM_KNL/SpMM_ASpT_SP.x ${ii}/${ii}.mtx
echo >> SpMM_KNL_SP.out
echo >> SpMM_KNL_SP_preprocessing.out
done


echo "dataset, MKL_GFLOPs(K=32), MKL_GFLOPs(K=128), TACO_GFLOPs(K=32), TACO_GFLOPs(K=128), CSB_GFLOPs(K=32), CSB_GFLOPs(K=128), ASpT_GFLOPs(K=32), ASpT_GFLOPs(K=128), ASpT_diff_%(K=32+K=128)" >> SpMM_KNL_DP.out
echo "dataset, preprocessing_ratio" >> SpMM_KNL_DP_preprocessing.out

for i in `ls -d */`
do
cd ${i}
ii=${i/\//}
cd ..
echo -n ${ii} >> SpMM_KNL_DP.out
echo -n "," >> SpMM_KNL_DP.out
echo -n ${ii} >> SpMM_KNL_DP_preprocessing.out
echo -n "," >> SpMM_KNL_DP_preprocessing.out
../SpMM_KNL/SpMM_MKL_DP.x --input=./${ii}/${ii}.mtx
../SpMM_KNL/SpMM_TACO_DP.x --input=./${ii}/${ii}.mtx
../CSB/spmm_d32 ${ii}/${ii}.mtx
../CSB/spmm_d128 ${ii}/${ii}.mtx
../SpMM_KNL/SpMM_ASpT_DP.x ${ii}/${ii}.mtx
echo >> SpMM_KNL_DP.out
echo >> SpMM_KNL_DP_preprocessing.out
done

echo "dataset, TACO_GFLOPs(K=32), TACO_GFLOPs(K=128), ASpT_GFLOPs(K=32), ASpT_GFLOPs(K=128), ASpT_diff_%(K=32+K=128)" >> SDDMM_KNL_SP.out
for i in `ls -d */`
do
cd ${i}
ii=${i/\//}
cd ..
echo -n ${ii} >> SDDMM_KNL_SP.out
echo -n "," >> SDDMM_KNL_SP.out
../SDDMM_KNL/SDDMM_TACO_SP.x --input=./${ii}/${ii}.mtx
../SDDMM_KNL/SDDMM_ASpT_SP.x ${ii}/${ii}.mtx
echo >> SDDMM_KNL_SP.out
done

echo "dataset, TACO_GFLOPs(K=32), TACO_GFLOPs(K=128), ASpT_GFLOPs(K=32), ASpT_GFLOPs(K=128), ASpT_diff_%(K=32+K=128)" >> SDDMM_KNL_DP.out
for i in `ls -d */`
do
cd ${i}
ii=${i/\//}
cd ..
echo -n ${ii} >> SDDMM_KNL_DP.out
echo -n "," >> SDDMM_KNL_DP.out
../SDDMM_KNL/SDDMM_TACO_DP.x --input=./${ii}/${ii}.mtx
../SDDMM_KNL/SDDMM_ASpT_DP.x ${ii}/${ii}.mtx
echo >> SDDMM_KNL_DP.out
done



mv *.out ../KNL_result
cd ..


