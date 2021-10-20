#ifndef _H_SHFL_FIX
#define _H_SHFL_FIX

#define __shfl __shfl2
#define __shfl_down __shfl_down2
#define __shfl_up __shfl_up2
#define __shfl_xor __shfl_xor2

template<typename T>
__device__ __forceinline__ T __shfl2(T var, int srcLane, int width=32) {
	return __shfl_sync(0xffffffff, var, srcLane, width);
}

template<typename T>
__device__ __forceinline__ T __shfl_down2(T var, int delta, int width=32) {
	return __shfl_down_sync(0xffffffff, var, delta, width);
}

template<typename T>
__device__ __forceinline__ T __shfl_up2(T var, int delta, int width=32) {
	return __shfl_up_sync(0xffffffff, var, delta, width);
}

template<typename T>
__device__ __forceinline__ T __shfl_xor2(T var, int laneMask, int width=32) {
	return __shfl_xor_sync(0xffffffff, var, laneMask, width);
}

#endif //_H_SHFL_FIX
