#ifndef CPU_CSR_KERNEL_H_
#define CPU_CSR_KERNEL_H_
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#ifdef profiling
#include "tools/ntimer.h"
#endif
#include <iostream>
#include "tools/qmalloc.h"
#include "tools/stats.h"
#include "tools/macro.h"

const int LEVEL1_DCACHE_LINESIZE = 64;
struct thread_data_t {
  QValue* x;
  bool* xb;
  QValue** xbp;
  int* index;
  char pad_data[LEVEL1_DCACHE_LINESIZE];
  void init(const int n) {
    x = (QValue*)qmalloc(n * sizeof(QValue) + LEVEL1_DCACHE_LINESIZE, __FUNCTION__, __LINE__);
    //xb = (bool*)qcalloc(n + LEVEL1_DCACHE_LINESIZE, sizeof(bool), __FUNCTION__, __LINE__);
    xb = (bool*)qmalloc(n + LEVEL1_DCACHE_LINESIZE, __FUNCTION__, __LINE__);
    index = (int*)qmalloc(n * sizeof(int) + LEVEL1_DCACHE_LINESIZE, __FUNCTION__, __LINE__);
    //memset(index, -1, n * sizeof(int) + LEVEL1_DCACHE_LINESIZE);
  }

  void init_memory_Managed(const int n) {
    xbp = (QValue**)qmalloc(n * sizeof(QValue*), __FUNCTION__, __LINE__);
    
    // xb = (bool*)qmalloc(n + LEVEL1_DCACHE_LINESIZE, __FUNCTION__, __LINE__);
    // index = (int*)qmalloc(n * sizeof(int) + LEVEL1_DCACHE_LINESIZE, __FUNCTION__, __LINE__);
    //memset(index, -1, n * sizeof(int) + LEVEL1_DCACHE_LINESIZE);
  }

  void init(QValue *x, bool* xb, int *index) {
    this->x = x;
    this->xb = xb;
    this->index = index;
  }

  thread_data_t() {
    x = NULL;
    xb = NULL;
    index = NULL;
    xbp = NULL;
  }

  thread_data_t(const int n) {
    init(n);
  }

  ~thread_data_t() {
    free(xb);
    free(x);
    free(index);
    xb = NULL;
    x = NULL;
    index = NULL;
    
  }
};

long long getSpMMFlops(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        const int m, const int k, const int n);

long spmmFootPrints(const int IA[], const int JA[],
    const int IB[], const int IC[],
    const int m, long *footPrintSum);

void sequential_CSR_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n);

void sequential_CSR_IC_nnzC(const int IA[], const int JA[],
        const int IB[], const int JB[],
        const int m, const int n, bool xb[],
        int* IC, int& nnzC);

void omp_CSR_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const int stride);
void omp_CSR_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride);
void noindex_somp_CSR_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const int stride);
thread_data_t* allocateThreadDatas(int nthreads, int n);
void freeThreadDatas(thread_data_t* thread_datas, int nthreads);
thread_data_t* allocateThreadDatas_memory_Managed(int nthreads, int n);
void freeThreadDatas_memory_Managed(thread_data_t* thread_datas, int nthreads);
void static_omp_CSR_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const int stride);
void static_omp_CSR_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride);
void static_omp_CSR_SpMM_memory_Managed(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const int stride);
void static_omp_CSR_SpMM_memory_Managed(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride);
void flops_omp_CSR_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const int stride);
void group_CSR_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const int stride);
void omp_CSR_IC_nnzC(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n, const thread_data_t& thread_data,
    int* IC, int& nnzC, const int stride);
void omp_CSR_IC_nnzC_Wrapper(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n, const thread_data_t thread_datas[],
    int* IC, int& nnzC, const int stride);
void dynamic_omp_CSR_IC_nnzC_footprints(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n, const thread_data_t& thread_data,
    int* IC, int& nnzC, int* footPrints, const int stride);
void static_omp_CSR_IC_nnzC(const int IA[], const int JA[],
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
int processCRowI(QValue x[], bool* xb,
    const int iAnnz, const int iJA[], const QValue iA[],
        const int IB[], const int JB[], const QValue B[],
        int* iJC, QValue* iC);

//extern int indexRowId;
//inline int indexProcessCRowI(int *restrict index, // index array must be initilized with -1
#ifdef __CUDACC__
inline int indexProcessCRowI(int *index,
    const int iAnnz, const int iJA[], const QValue iA[],
    const int IB[], const int JB[], const QValue B[],
    int* iJC, QValue* iC) {
#else
inline int indexProcessCRowI(int *restrict index,
    const int iAnnz, const int iJA[], const QValue iA[],
    const int IB[], const int JB[], const QValue B[],
    int* restrict iJC, QValue* restrict iC) {
#endif
  if (iAnnz == 0) {
    return 0;
  }
  int ip = -1;
  int jp = 0;
  int j = iJA[jp];
  for(int tp = IB[j]; tp < IB[j + 1]; ++tp) {
    int t = JB[tp];
    iJC[++ip] = t;
    index[t] = ip;
    iC[ip] = iA[jp] * B[tp];
    // if (indexRowId == 3 && t == 247)
    //   printf("C[%d][%d]+=A[%d][%d]*B[%d][%d]=%f*%f=%f\n", indexRowId, t, indexRowId, j, j, t, iA[jp], B[tp], iC[index[t]]);
  }
  for(int jp = 1; jp < iAnnz; ++jp) {
    int j = iJA[jp];
#pragma unroll(2)
    for(int tp = IB[j]; tp < IB[j + 1]; ++tp) {
      int t = JB[tp];
      if(index[t] == -1) {
        iJC[++ip] = t;
        index[t] = ip;
        iC[ip] = iA[jp] * B[tp];
      } else {
        iC[index[t]] += iA[jp] * B[tp];
      }
      // if (indexRowId == 3 && t == 247)
      //   printf("C[%d][%d]+=A[%d][%d]*B[%d][%d]=%f*%f=%f\n", indexRowId, t, indexRowId, j, j, t, iA[jp], B[tp], iC[index[t]]);
      // This hack will remove if condition but it will make program slightly slow due to more operations.
      // This may worth a try on Xeon Phi machines.
      // int f = index[t] >> 31;
      // ip += f & 1;
      // index[t] += f & (ip + 1);
      // iJC[index[t]] = t;
      // iC[index[t]] += iA[jp] * B[tp];
    }
  }
  ++ip;
  for(int vp = 0; vp < ip; ++vp) {
    int v = iJC[vp];
    index[v] = -1;
  }
  return ip;
}

void omp_CSR_RMCL_OneStep(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride);
void static_omp_CSR_RMCL_OneStep(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride);
void static_fair_CSR_RMCL_OneStep(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const int stride);
void hybrid_omp_CSR_RMCL_OneStep(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride);
inline void matrix_relocation(const int rowsNnz[], const int m,
        int* &IC, int* &JC, QValue* &C, int& nnzC) {
#ifdef profiling
    QValue rnow = time_in_mill_now();
#endif
    int top = rowsNnz[0];
    for (int i = 1; i < m; ++i) {
      int up = IC[i] + rowsNnz[i];
      const int preTop = top;
      for (int j = IC[i]; j < up; ++j) {
        JC[top] = JC[j];
        C[top++] = C[j];
      }
      IC[i] = preTop;
    }
    IC[m] = top;
    nnzC = top;
    JC = (int*)realloc(JC, sizeof(int) * nnzC);
    C = (QValue*)realloc(C, sizeof(QValue) * nnzC);
#ifdef profiling
    printf("time passed for seq relocate IC, JC and C %lf with nnzC=%d\n", time_in_mill_now() - rnow, nnzC);
#endif
}

//This function must be called in OpenMP parallel region
void omp_matrix_relocation(int rowsNnz[], const int m, const int tid, const int stride,
        int* &IC, int* &JC, QValue* &C, int& nnzC);

inline int cRowiCount(const int i, const int IA[], const int JA[], const int IB[], const int JB[], int iJC[], bool xb[]) {
  if (IA[i] == IA[i + 1]) {
    return 0;
  }
  int count = -1;
  int vp = IA[i];
  int v = JA[vp];
  for (int kp = IB[v]; kp < IB[v+1]; ++kp) {
    int k = JB[kp];
    iJC[++count] = k;
    xb[k] = true;
  }
  for (int vp = IA[i] + 1; vp < IA[i + 1]; ++vp) {
    int v = JA[vp];
    for (int kp = IB[v]; kp < IB[v+1]; ++kp) {
      int k = JB[kp];
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

void dynamic_omp_CSR_flops(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n,
    long* rowFlops, const int stride);

#endif
