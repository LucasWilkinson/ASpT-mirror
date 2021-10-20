#ifndef CPU_CSC_KERNEL_H_
#define CPU_CSC_KERNEL_H_
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "cpu_csr_kernel.h"
#ifdef profiling
#include "tools/ntimer.h"
#endif
#include <iostream>
#include "tools/qmalloc.h"
#include "tools/stats.h"
#include "tools/macro.h"

// const int LEVEL1_DCACHE_LINESIZE = 64;
// struct thread_data_t {
//   QValue* x;
//   bool* xb;
//   QValue** xbp;
//   int* index;
//   char pad_data[LEVEL1_DCACHE_LINESIZE];
//   void init(const int m) {
//     x = (QValue*)qmalloc(m * sizeof(QValue) + LEVEL1_DCACHE_LINESIZE, __FUNCTION__, __LINE__);
//     //xb = (bool*)qcalloc(n + LEVEL1_DCACHE_LINESIZE, sizeof(bool), __FUNCTION__, __LINE__);
//     xb = (bool*)qmalloc(m + LEVEL1_DCACHE_LINESIZE, __FUNCTION__, __LINE__);
//     index = (int*)qmalloc(m * sizeof(int) + LEVEL1_DCACHE_LINESIZE, __FUNCTION__, __LINE__);
//     //memset(index, -1, n * sizeof(int) + LEVEL1_DCACHE_LINESIZE);
//   }

//   void init_memory_Managed(const int m) {
//     xbp = (QValue**)qmalloc(m * sizeof(QValue*), __FUNCTION__, __LINE__);
//   }

//   void init(QValue *x, bool* xb, int *index) {
//     this->x = x;
//     this->xb = xb;
//     this->index = index;
//   }

//   thread_data_t() {
//     x = NULL;
//     xb = NULL;
//     index = NULL;
//     xbp = NULL;
//   }

//   thread_data_t(const int m) {
//     init(m);
//   }

//   ~thread_data_t() {
//     free(xb);
//     free(x);
//     free(index);
//     xb = NULL;
//     x = NULL;
//     index = NULL;
    
//   }
// };

long long getCSCSpMMFlops(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        const int m, const int k, const int n);

long spmmFootPrints(const int IB[], const int JB[],
    const int IA[], const int IC[],
    const int n, long *footPrintSum);

void sequential_CSC_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n);

void sequential_CSC_IC_nnzC(const int IA[], const int JA[],
        const int IB[], const int JB[],
        const int m, const int n, bool xb[],
        int* IC, int& nnzC);

// void omp_CSC_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
//         const int IB[], const int JB[], const QValue B[], const int nnzB,
//         int* &IC, int* &JC, QValue* &C, int& nnzC,
//         const int m, const int k, const int n, const int stride);
// void omp_CSC_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
//         const int IB[], const int JB[], const QValue B[], const int nnzB,
//         int* &IC, int* &JC, QValue* &C, int& nnzC,
//         const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride);
// void noindex_somp_CSC_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
//         const int IB[], const int JB[], const QValue B[], const int nnzB,
//         int* &IC, int* &JC, QValue* &C, int& nnzC,
//         const int m, const int k, const int n, const int stride);
// thread_data_t* allocateThreadDatas(int nthreads, int m);
// void freeThreadDatas(thread_data_t* thread_datas, int nthreads);
// thread_data_t* allocateThreadDatas_memory_Managed(int nthreads, int m);
//void freeThreadDatas_memory_Managed(thread_data_t* thread_datas, int nthreads);
void static_omp_CSC_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const int stride);
void static_omp_CSC_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride);
// void static_omp_CSC_SpMM_memory_Managed(const int IA[], const int JA[], const QValue A[], const int nnzA,
//         const int IB[], const int JB[], const QValue B[], const int nnzB,
//         int* &IC, int* &JC, QValue* &C, int& nnzC,
//         const int m, const int k, const int n, const int stride);
// void static_omp_CSC_SpMM_memory_Managed(const int IA[], const int JA[], const QValue A[], const int nnzA,
//         const int IB[], const int JB[], const QValue B[], const int nnzB,
//         int* &IC, int* &JC, QValue* &C, int& nnzC,
//         const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride);
// void flops_omp_CSC_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
//         const int IB[], const int JB[], const QValue B[], const int nnzB,
//         int* &IC, int* &JC, QValue* &C, int& nnzC,
//         const int m, const int k, const int n, const int stride);
// void group_CSC_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
//         const int IB[], const int JB[], const QValue B[], const int nnzB,
//         int* &IC, int* &JC, QValue* &C, int& nnzC,
//         const int m, const int k, const int n, const int stride);
void omp_CSC_IC_nnzC(const int IA[], const int JA[],
     const int IB[], const int JB[],
     const int m, const int n, const thread_data_t& thread_data,
     int* IC, int& nnzC, const int stride);
void omp_CSC_IC_nnzC_Wrapper(const int IA[], const int JA[],
     const int IB[], const int JB[],
     const int m, const int n, const thread_data_t thread_datas[],
     int* IC, int& nnzC, const int stride);
void dynamic_omp_CSC_IC_nnzC_footprints(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n, const thread_data_t& thread_data,
    int* IC, int& nnzC, int* footPrints, const int stride);
void static_omp_CSC_IC_nnzC(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n, const thread_data_t& thread_data,
    int* IC, int& nnzC, const int stride, const int ends[], const int tid);
/*void processCRowI(
    //x and xb are used for temp use only and will have the same value when back.
    //xb must be all zeros before calling this functions.
    QValue x[], bool* xb,
    //IAi and IAi1 are starting and ending postions of A's row i in JA array.
    const int IAi, const int IAi1, const int JA[], const QValue A[],
        const int IB[], const int JB[], const QValue B[],
        const int ICi, int* JC, QValue* C);*/
int processCColI(QValue x[], bool* xb,
    const int iBnnz, const int iJB[], const QValue iB[],
    const int IA[], const int JA[], const QValue A[],
    int* iJC, QValue* iC);


#ifdef __CUDACC__
inline int indexProcessCColI(int *index,
    const int iBnnz, const int iJB[], const QValue iB[],
    const int IA[], const int JA[], const QValue A[],
    int* iJC, QValue* iC);
#else
inline int indexProcessCColI(int *restrict index,
    const int iBnnz, const int iJB[], const QValue iB[],
    const int IA[], const int JA[], const QValue A[],
    int* restrict iJC, QValue* restrict iC) {
#endif
  if (iBnnz == 0) {
    return 0;
  }
  int ip = -1;
  int jp = 0;
  int j = iJB[jp];
  for(int tp = IA[j]; tp < IA[j + 1]; ++tp) {
    int t = JA[tp];
    iJC[++ip] = t;
    index[t] = ip;
    iC[ip] = iB[jp] * A[tp];
   
  }
  for(int jp = 1; jp < iBnnz; ++jp) {
    int j = iJB[jp];
#pragma unroll(2)
    for(int tp = IA[j]; tp < IA[j + 1]; ++tp) {
      int t = JA[tp];
      if(index[t] == -1) {
        iJC[++ip] = t;
        index[t] = ip;
        iC[ip] = iB[jp] * A[tp];
      } else {
        iC[index[t]] += iB[jp] * A[tp];
      }
      
    }
  }
  ++ip;
  for(int vp = 0; vp < ip; ++vp) {
    int v = iJC[vp];
    index[v] = -1;
  }
  return ip;
}

inline int cColiCount(const int i, const int IA[], const int JA[], const int IB[], const int JB[], int iJC[], bool xb[]) {
  if (IB[i] == IB[i + 1]) {
    return 0;
  }
  int count = -1;
  int vp = IB[i];
  int v = JB[vp];
  for (int kp = IA[v]; kp < IA[v+1]; ++kp) {
    int k = JA[kp];
    iJC[++count] = k;
    xb[k] = true;
  }
  for (int vp = IB[i] + 1; vp < IB[i + 1]; ++vp) {
    int v = JB[vp];
    for (int kp = IA[v]; kp < IA[v+1]; ++kp) {
      int k = JA[kp];
      if(xb[k] == false) {
        iJC[++count] = k;
        xb[k] = true;
      }
    }
  }
  ++count;
  for(int jp = 0; jp < count; ++jp) {
    int j = iJC[jp];
    xb[j] = false;
  }
  return count;
}


#endif
