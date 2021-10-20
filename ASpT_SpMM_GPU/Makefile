NVCC=nvcc
CUDA_ARCH_GENCODE="arch=compute_86,code=sm_86"
NVCC_ARGS=

clean:
	rm dspmm_128 dspmm_32 sspmm_128 sspmm_32

%: %.cu
	$(NVCC) -std=c++14 -Xcompiler -rdynamic -lineinfo -g -G --generate-line-info  -O3 $(NVCC_ARGS) -gencode $(CUDA_ARCH_GENCODE) $< --use_fast_math -Xptxas "-v -dlcm=ca" -o $@

# nvcc -std=c++14 -Xcompiler -rdynamic -lineinfo -g -G --generate-line-info  -O3 ${NVCC_ARGS} -gencode ${CUDA_ARCH_GENCODE} sspmm_32.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o sspmm_32
# nvcc -std=c++14 -Xcompiler -rdynamic -lineinfo -g -G --generate-line-info  -O3 ${NVCC_ARGS} -gencode ${CUDA_ARCH_GENCODE} sspmm_128.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o sspmm_128
# nvcc -std=c++14 -Xcompiler -rdynamic -lineinfo -g -G --generate-line-info  -O3 ${NVCC_ARGS} -gencode ${CUDA_ARCH_GENCODE} dspmm_32.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o dspmm_32
# nvcc -std=c++14 -Xcompiler -rdynamic -lineinfo -g -G --generate-line-info  -O3 ${NVCC_ARGS} -gencode ${CUDA_ARCH_GENCODE} dspmm_128.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o dspmm_128