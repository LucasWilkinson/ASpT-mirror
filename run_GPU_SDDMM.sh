#!/bin/bash
mkdir GPU_SDDMM_result
cp tr.sh ./BIDMach/scripts
pp=`pwd`
cd data
rm SDDMM_GPU_SP.out


echo "dataset, ASpT_GFLOPs(K=32), ASpT_diff_%(K=32), ASpT_GFLOPs(K=128), ASpT_diff_%(K=128), BIDMach_GFLOPs(K=32), BIDMach_GFLOPs(K=128)" >> SDDMM_GPU_SP.out

for i in `ls -d */`
do
cd ${i}
ii=${i/\//}
cd ..
echo -n ${ii} >> SDDMM_GPU_SP.out
echo -n "," >> SDDMM_GPU_SP.out
../ASpT_SDDMM_GPU/sddmm_32 ${ii}/${ii}.mtx 32
../ASpT_SDDMM_GPU/sddmm_128 ${ii}/${ii}.mtx 128

../BIDMach/scripts/tr.sh ${pp} ${ii} 32 
../BIDMach/bidmach alp.ssc &> tmp
cat tmp | grep TTAAGG &> tmp2
sed -i "1s/TTAAGG,//g" tmp2
cat tmp2 | tr "\n" "," >> SDDMM_GPU_SP.out

../BIDMach/scripts/tr.sh ${pp} ${ii} 128
../BIDMach/bidmach alp.ssc &> tmp
cat tmp | grep TTAAGG &> tmp2
sed -i "1s/TTAAGG,//g" tmp2
cat tmp2 | tr "\n" "," >> SDDMM_GPU_SP.out

echo >> SDDMM_GPU_SP.out
done

mv SDDMM_GPU_SP.out ../GPU_SDDMM_result
