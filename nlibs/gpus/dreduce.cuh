
/* This version uses contiguous threads, but its interleaved
   addressing results in many shared memory bank conflicts.
 */
template <class T, int BLOCK_THREADS>
__device__ void
reduce0(const T *g_idata, T *g_odata, const unsigned count) {
  __shared__ T sdata[BLOCK_THREADS];
  if (threadIdx.x == 0) {
    *g_odata = 0;
  }
  unsigned tid = threadIdx.x;
  const unsigned upcount = (count + blockDim.x - 1) / blockDim.x * blockDim.x;
  for (unsigned i = threadIdx.x; i < upcount; i += blockDim.x) {
    sdata[tid] = (i < count) ? g_idata[i] : 0.0;
    __syncthreads();
    // do reduction in shared mem
    for (unsigned s = 1; s < blockDim.x; s *= 2) {
      if ((tid % (2*s)) == 0) {
        sdata[tid] += sdata[tid + s];
        //printf("sdata[%d] + sdata[%d] = %e\n", tid, tid + s, sdata[tid]);
      }
      __syncthreads();
    }
  // write result for this block to global mem
    if (tid == 0) {
      g_odata[0] += sdata[0];
    }
  }
}

/*
This version uses sequential addressing -- no divergence or bank conflicts.
*/
template <class T, int BLOCK_THREADS>
__device__ void
reduce2(const T *g_idata, T *g_odata, const unsigned count) {
  __shared__ T sdata[BLOCK_THREADS];
  if (threadIdx.x == 0) {
    *g_odata = 0;
  }
  unsigned tid = threadIdx.x;
  const unsigned upcount = (count + blockDim.x - 1) / blockDim.x * blockDim.x;
  for (unsigned i = threadIdx.x; i < upcount; i += blockDim.x) {
    sdata[tid] = (i < count) ? g_idata[i] : 0.0;
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) {
        sdata[tid] += sdata[tid + s];
      }
      __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) {
      g_odata[0] += sdata[0];
    }
  }
}
