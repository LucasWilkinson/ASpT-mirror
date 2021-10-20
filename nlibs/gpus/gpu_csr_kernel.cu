#include "gpus/cuda_handle_error.h"
#include "gpus/cusparse_spmm.h"
#include "gpus/dutil.cuh"
#include "gpus/gpu_csr_kernel.h"
#include "gpus/timer.h"
#include "tools/ntimer.h"
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/remove.h>

__global__ void outputCSRKernel(const int *rowPtr, const int *colInd, const QValue *values, int rows) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (int i = 0; i < rows; i++) {
      for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
        int col = colInd[j];
        double val = values[j];
        printf("%d\t%d\t%.6lf\n", i, col, val);
      }
    }
  }
  printf("rowPtr= ");
  for (int i = 0; i < rows + 1; i++) {
    printf("%d ", rowPtr[i]);
  }
  printf("\ncolInd, values= ");
  for (int i = 0; i < rowPtr[rows]; i++) {
    printf("<%d, %lf>", colInd[i], values[i]);
  }
  printf("\ndone\n");
}

void gpuOutputCSRWrapper(const CSR dA, const char* msg) {
  printf("%s\n", msg);
  printf("rows=%d cols=%d nnz=%d rowPtr=%p colInd=%p values=%p\n", dA.rows, dA.cols, dA.nnz,
      dA.rowPtr, dA.colInd, dA.values);
  outputCSRKernel<<<1, 1>>>(dA.rowPtr, dA.colInd, dA.values, dA.rows);
  cudaDeviceSynchronize();
}

__global__ void gpu_CSR_IC_nnzC(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n,
    int* IC,
    bool *xbs, int *iJCs) {
  bool *xb = xbs + blockIdx.x * n;
  int *iJC = iJCs + blockIdx.x * n;
  if (threadIdx.x == 0 && blockIdx.x ==0) {
    IC[m] = 0;
  }
  __shared__ int count;
  if (threadIdx.x == 0) {
    count = 0;
  }
  __syncthreads();
  for (int i = blockIdx.x; i < m; i += gridDim.x) {
    IC[i] = 0;
    for (int vp = IA[i]; vp < IA[i + 1]; ++vp) {
      int v = JA[vp];
      for (int kp = IB[v] + threadIdx.x; kp < IB[v + 1]; kp += blockDim.x) {
        int k = JB[kp];
        if (xb[k] == false) {
          iJC[atomicAdd(&count, 1)] = k;
          xb[k] = true;
        }
      }
      __syncthreads();
    }
    for (int jp = threadIdx.x; jp < count; jp += blockDim.x) {
      int j = iJC[jp];
      xb[j] = false;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      IC[i] = count;
      count = 0;
    }
  }
}

template <int BLOCK_THREADS>
__global__ void gpuSpMMKernel(const int *IA, const int *JA, const QValue *A,
    const int *IB, const int *JB, const QValue *B,
    const int *IC, int *JC, QValue *C,
    bool *xbs, QValue *xs,
    const int m, const int k, const int n) {
  __shared__ int dcount;
  if (threadIdx.x == 0) {
    dcount = 0;
  }
  bool *xb = xbs + blockIdx.x * n;
  QValue *x = xs + blockIdx.x * n;
  __syncthreads();
  for (int i = blockIdx.x; i < m; i += gridDim.x) {
    const int ICi = IC[i];
    int *iJC = JC + ICi;
    for (int jp = IA[i]; jp < IA[i + 1]; ++jp) {
      int j = JA[jp];
      const QValue Ajp = A[jp];
      for (int tp = IB[j] + threadIdx.x; tp < IB[j + 1]; tp += blockDim.x) {
        int t = JB[tp];
        if (xb[t] == false) {
          iJC[atomicAdd(&dcount, 1)] = t;
          xb[t] = true;
          x[t] = Ajp * B[tp];
        } else {
          x[t] += Ajp * B[tp];
        }
      }
      __syncthreads();
    }
    for (int jp = threadIdx.x; jp < dcount; jp += blockDim.x) {
      int j = iJC[jp];
      C[jp + ICi] = x[j];
      x[j] = 0.0;
      xb[j] = false;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      dcount = 0;
    }
  }
}

CSR gpuSpMMWrapper(const CSR &dA, const CSR &dB) {
  timer t;
  CSR dC;
  //const int NBLOCKS = 1; const int NTHREADS = 1;
  const int NBLOCKS = 512; const int NTHREADS = 128;
  int m = dA.rows;
  int k = dA.cols;
  int n = dB.cols;
  bool *xbs = NULL;
  int *iJCs = NULL;
  QValue *xs = NULL;
  HANDLE_ERROR(cudaMalloc((void**)&xbs, NBLOCKS * n * sizeof(bool))); cudaMemset(xbs, 0, NBLOCKS * n * sizeof(bool));
  HANDLE_ERROR(cudaMalloc((void**)&xs, NBLOCKS * n * sizeof(QValue))); cudaMemset(xs, 0, NBLOCKS * n * sizeof(QValue));
  HANDLE_ERROR(cudaMalloc((void**)&iJCs, NBLOCKS * n * sizeof(int)));

  HANDLE_ERROR(cudaMalloc((void**)&dC.rowPtr, (m + 1) * sizeof(int)));
  timer t2;
  gpu_CSR_IC_nnzC<<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dB.rowPtr, dB.colInd,
      m, n, dC.rowPtr, xbs, iJCs);
  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaFree(iJCs));
  thrust::device_ptr<int> dIC = thrust::device_pointer_cast(dC.rowPtr);
  thrust::exclusive_scan(dIC, dIC + m + 1, dIC);
  double nnzTime = t.milliseconds_elapsed();
  int cNnz = dIC[m];
  int hh = dIC[1];
  //printf("dIC[1] = %d\n", hh);
  HANDLE_ERROR(cudaMalloc((void**)&dC.colInd, cNnz * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&dC.values, cNnz * sizeof(QValue)));
  gpuSpMMKernel<NTHREADS><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dA.values,
      dB.rowPtr, dB.colInd, dB.values,
      dC.rowPtr, dC.colInd, dC.values,
      xbs, xs,
      m, k, n);
  cudaDeviceSynchronize();
  HANDLE_ERROR(cudaGetLastError());
  //hh = dIC[1]; printf("dIC[1] = %d\n", hh);
  dC.rows = m;
  dC.cols = n;
  dC.nnz = cNnz;
  HANDLE_ERROR(cudaFree(xbs));
  HANDLE_ERROR(cudaFree(xs));
  double timeUsed = t.milliseconds_elapsed();
  printf("Time used for gpu spmm %lf nnzTime=%lf\n", timeUsed, nnzTime);
  return dC;
}

template <int BLOCK_THREADS>
__global__ void gpuRmclOneStepKernel(const int *IA, const int *JA, const QValue *A,
    const int *IB, const int *JB, const QValue *B,
    int *IC, int *JC, QValue *C,
    bool *xbs, QValue *xs,
    const int m, const int k, const int n) {
  __shared__ int dcount;
  if (threadIdx.x == 0) {
    dcount = 0;
  }
  bool *xb = xbs + blockIdx.x * n;
  QValue *x = xs + blockIdx.x * n;
  __syncthreads();
  for (int i = blockIdx.x; i < m; i += gridDim.x) {
    const int ICi = IC[i];
    int *iJC = JC + ICi;
    for (int jp = IA[i]; jp < IA[i + 1]; ++jp) {
      int j = JA[jp];
      const QValue Ajp = A[jp];
      for (int tp = IB[j] + threadIdx.x; tp < IB[j + 1]; tp += blockDim.x) {
        int t = JB[tp];
        if (xb[t] == false) {
          iJC[atomicAdd(&dcount, 1)] = t;
          xb[t] = true;
          x[t] = Ajp * B[tp];
        } else {
          x[t] += Ajp * B[tp];
        }
      }
      __syncthreads();
    }
    for (int jp = threadIdx.x; jp < dcount; jp += blockDim.x) {
      int j = iJC[jp];
      C[jp + ICi] = x[j];
      x[j] = 0.0;
      xb[j] = false;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      /*if (i < 10) {
        printf("i=%d dcount=%d\n", i, dcount);
      }*/
    }
    double thresh = inflationR2Thresh<BLOCK_THREADS>(dcount, C + ICi);
    __syncthreads();
    int rowPruneCount = threshPruneNormalize<BLOCK_THREADS>(dcount, thresh, JC + ICi, C + ICi);
    __syncthreads();
    if (threadIdx.x == 0) {
      IC[i] = rowPruneCount;
      /*if (i < 10) {
        printf("i=%d rowPruneCount=%d thresh=%e dcount=%d\n", i, rowPruneCount, thresh, dcount);
      }*/
      dcount = 0;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && blockIdx.x ==0) {
    IC[m] = 0;
  }
}

struct is_minus_one {
  __host__ __device__
  bool operator() (const int x) {
    return (x == -1);
  }
};

template <int NBLOCKS, int NTHREADS>
CSR gpuRmclOneStepWrapper(const CSR &dA, const CSR &dB,
    bool *xbs, QValue *xs, int *iJCs) {
  CSR dC;
  int m = dA.rows;
  int k = dA.cols;
  int n = dB.cols;
  //timer t;
  HANDLE_ERROR(cudaMalloc((void**)&dC.rowPtr, (m + 1) * sizeof(int)));
  gpu_CSR_IC_nnzC<<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dB.rowPtr, dB.colInd,
      m, n, dC.rowPtr, xbs, iJCs);
  HANDLE_ERROR(cudaGetLastError());
  thrust::device_ptr<int> dIC = thrust::device_pointer_cast(dC.rowPtr);
  thrust::exclusive_scan(dIC, dIC + m + 1, dIC);
  int cNnz = dIC[m];
  HANDLE_ERROR(cudaMalloc((void**)&dC.colInd, cNnz * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&dC.values, cNnz * sizeof(QValue)));
  gpuRmclOneStepKernel<NTHREADS><<<NBLOCKS, NTHREADS>>>(dA.rowPtr, dA.colInd, dA.values,
      dB.rowPtr, dB.colInd, dB.values,
      dC.rowPtr, dC.colInd, dC.values,
      xbs, xs,
      m, k, n);
  thrust::remove_if(thrust::device_pointer_cast(dC.values),
      thrust::device_pointer_cast(dC.values + cNnz),
      thrust::device_pointer_cast(dC.colInd), is_minus_one());
  thrust::remove(thrust::device_pointer_cast(dC.colInd),
      thrust::device_pointer_cast(dC.colInd + cNnz), -1);
  thrust::exclusive_scan(dIC, dIC + m + 1, dIC);
  cudaDeviceSynchronize();
  HANDLE_ERROR(cudaGetLastError());
  dC.rows = m;
  dC.cols = n;
  dC.nnz = dIC[m];
  printf("dCnnz=%d cNnz=%d\n", dC.nnz, cNnz);
  //double timeUsed = t.milliseconds_elapsed();
  return dC;
}

void gpuRmclIter(const int maxIter, const CSR Mgt, CSR &Mt) {
  //const int NBLOCKS = 1; const int NTHREADS = 2;
  const int NBLOCKS = 512; const int NTHREADS = 128;
  int n = Mt.cols;
  CSR dMgt = Mgt.toGpuCSR();
  CSR dMt = Mt.toGpuCSR();
  bool *xbs = NULL;
  int *iJCs = NULL;
  QValue *xs = NULL;
  HANDLE_ERROR(cudaMalloc((void**)&xbs, NBLOCKS * n * sizeof(bool))); cudaMemset(xbs, 0, NBLOCKS * n * sizeof(bool));
  HANDLE_ERROR(cudaMalloc((void**)&xs, NBLOCKS * n * sizeof(QValue))); //cudaMemset(xs, 0, NBLOCKS * n * sizeof(double));
  HANDLE_ERROR(cudaMalloc((void**)&iJCs, NBLOCKS * n * sizeof(int)));
  double now = time_in_mill_now();
  for (int iter = 0; iter < maxIter; ++iter) {
    timer t;
    CSR dNewMt = gpuRmclOneStepWrapper<NBLOCKS, NTHREADS>(dMgt, dMt, xbs, xs, iJCs);
    dMt.deviceDispose();
    dMt = dNewMt;
    printf("iter %d done in %lf milliseconds\n", iter, t.milliseconds_elapsed());
  }
  cudaDeviceSynchronize();
  cout << "gpu iter finish in " << time_in_mill_now() - now << "\n";
  now = time_in_mill_now();
  Mt.dispose();
  Mt = dMt.toCpuCSR();
  HANDLE_ERROR(cudaFree(iJCs));
  HANDLE_ERROR(cudaFree(xbs));
  HANDLE_ERROR(cudaFree(xs));
  cudaDeviceSynchronize();
  cout << "gpuRmclIter with copy back from device to host and cudaFree finish in " << time_in_mill_now() - now << "\n";
}
