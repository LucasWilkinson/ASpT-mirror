#!/bin/bash
mkdir GPU_SpMM_result

cd data
rm SpMM_GPU_SP.out
rm SpMM_GPU_DP.out
rm SpMM_GPU_SP_preprocessing.out
rm SpMM_GPU_DP_preprocessing.out

echo "dataset, cuSPARSE_GFLOPs(K=32), cuSPARSE(K=128)_GFLOPs, ASpT_GFLOPs(K=32), ASpT_diff_%(K=32), ASpT_GFLOPs(K=128), ASpT_diff_%(K=128), merge_SpMM_GFLOPs(K=32), merge_SpMM_GFLOPs(K=128)" >> SpMM_GPU_SP.out
echo "dataset, preprocessing_ratio" >> SpMM_GPU_SP_preprocessing.out

for i in `ls -d */`
do
cd ${i}
ii=${i/\//}
cd ..
echo -n ${ii} >> SpMM_GPU_SP.out
echo -n "," >> SpMM_GPU_SP.out
echo -n ${ii} >> SpMM_GPU_SP_preprocessing.out
echo -n "," >> SpMM_GPU_SP_preprocessing.out
../cuSPARSE_SpMM/cuSPARSE_SP ${ii}/${ii}.mtx
../ASpT_SpMM_GPU/sspmm_32 ${ii}/${ii}.mtx 32
../ASpT_SpMM_GPU/sspmm_128 ${ii}/${ii}.mtx 128
../merge-spmm/bin/gbspmm  --tb=128 --nt=32 --max_ncols=32 --iter=1 ${ii}/${ii}.mtx
../merge-spmm/bin/gbspmm  --tb=128 --nt=32 --max_ncols=128 --iter=1 ${ii}/${ii}.mtx
echo >> SpMM_GPU_SP.out
echo >> SpMM_GPU_SP_preprocessing.out
done

echo "dataset, cuSPARSE_GFLOPs(K=32), cuSPARSE(K=128)_GFLOPs, ASpT_GFLOPs(K=32), ASpT_diff_%(K=32), ASpT_GFLOPs(K=128), ASpT_diff_%(K=128)" >> SpMM_GPU_DP.out
echo "dataset, preprocessing_ratio" >> SpMM_GPU_DP_preprocessing.out

for i in `ls -d */`
do
cd ${i}
ii=${i/\//}
cd ..
echo -n ${ii} >> SpMM_GPU_DP.out
echo -n "," >> SpMM_GPU_DP.out
echo -n ${ii} >> SpMM_GPU_DP_preprocessing.out
echo -n "," >> SpMM_GPU_DP_preprocessing.out
../cuSPARSE_SpMM/cuSPARSE_DP ${ii}/${ii}.mtx
../ASpT_SpMM_GPU/dspmm_32 ${ii}/${ii}.mtx 32
../ASpT_SpMM_GPU/dspmm_128 ${ii}/${ii}.mtx 128
echo >> SpMM_GPU_DP.out
echo >> SpMM_GPU_DP_preprocessing.out
done

rm gmon.out
mv *.out ../GPU_SpMM_result
cd ..

#for i in `ls -d *.mtx`
#do
#../ASpT_SpMM/sddmm_32 ${i} 32
#../ASpT_SpMM/sddmm_128 ${i} 128
#done


