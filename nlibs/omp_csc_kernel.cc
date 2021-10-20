#include <omp.h>
#include <iostream>
#include "tools/ntimer.h"
#include "cpu_csc_kernel.h"
#include "tools/util.h"
#include "tools/prefixSum.h"
using namespace std;

#ifdef USE_MIC
#define AGGR
#endif

// thread_data_t* allocateThreadDatas(int nthreads, int m) {
// #ifdef profiling
//   QValue now = time_in_mill_now();
// #endif
//   thread_data_t* thread_datas = (thread_data_t*)malloc(nthreads * sizeof(thread_data_t));
// #ifdef AGGR
//   QValue *xs = (QValue*)qmalloc((m * sizeof(QValue) + LEVEL1_DCACHE_LINESIZE) * nthreads, __FUNCTION__, __LINE__);
//   bool *xbs = (bool*)qmalloc((m + LEVEL1_DCACHE_LINESIZE) * nthreads, __FUNCTION__, __LINE__);
//   int *indices = (int*)qmalloc((m * sizeof(int) + LEVEL1_DCACHE_LINESIZE) * nthreads, __FUNCTION__, __LINE__);
  
// #endif
//   for(int i = 0; i < nthreads; i++) {
// #ifndef AGGR
//     thread_datas[i].init(m);
// #else
//     QValue *x = (QValue*)((char*)xs + i * (m * sizeof(QValue) + LEVEL1_DCACHE_LINESIZE));
//     //bool *xb = (bool*)((char*)xbs + i * (m + LEVEL1_DCACHE_LINESIZE));
//     bool *xb = (bool*)x;
//     int *index = (int*)((char*)indices + i * (m * sizeof(int) + LEVEL1_DCACHE_LINESIZE));
//     thread_datas[i].init(x, xb, index);
// #endif
//   }
// #ifdef profiling
//   printf("Time passed for %s in %lf milliseconds\n", __func__, time_in_mill_now() - now);
// #endif
//   return thread_datas;
// }

// void freeThreadDatas(thread_data_t* thread_datas, int nthreads) {
// #ifdef profiling
//   QValue now = time_in_mill_now();
// #endif
// #ifndef AGGR
//   for(int i = 0; i < nthreads; i++) {
//     thread_datas[i].~thread_data_t();
//   }
// #else
//   free(thread_datas[0].x);
//   //free(thread_datas[0].xb);
//   free(thread_datas[0].index);
//   free(thread_datas);
// #endif
// #ifdef profiling
//   printf("Time passed for %s in %lf milliseconds\n", __func__, time_in_mill_now() - now);
// #endif
// }


void omp_CSC_IC_nnzC_Wrapper(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n, const thread_data_t thread_datas[],
    int* IC, int& nnzC, const int stride) {
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    omp_CSC_IC_nnzC(IA, JA, IB, JB,
    m, n, thread_datas[tid],
    IC, nnzC, stride);
  }
}

/*
 * static_omp_CSC_IC_nnzC reminder: this function must be called in #pragma omp parallel regions
 * to parallelly execution.
 * */
void static_omp_CSC_IC_nnzC(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n, const thread_data_t& thread_data,
    int* IC, int& nnzC, const int stride, const int ends[], const int tid) {
  int *iJC = (int*)thread_data.index;
  bool *xb = thread_data.xb;
  memset(xb, 0, m);
#pragma omp barrier
  int low = ends[tid];
  int high = ends[tid + 1];
  for (int i = low; i < high; ++i) {
    IC[i] = cColiCount(i, IA, JA, IB, JB, iJC, xb);
  }
#pragma omp barrier
  noTileOmpPrefixSum(IC, IC, n);
#pragma omp single
  {
    nnzC = IC[n];
  }
}

/*
 * omp_CSC_IC_nnzC reminder: this function must be called in #pragma omp parallel regions
 * to parallelly execution.
 * */
void omp_CSC_IC_nnzC(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n, const thread_data_t& thread_data,
    int* IC, int& nnzC, const int stride) {
  int *iJC = (int*)thread_data.index;
  bool *xb = thread_data.xb;
  memset(xb, 0, m);
#pragma omp for schedule(dynamic)
  for (int it = 0; it < n; it += stride) {
    int up = it + stride < n ? it + stride : n;
    for (int i = it; i < up; ++i) {
      IC[i] = cColiCount(i, IA, JA, IB, JB, iJC, xb);
    }
  }
#pragma omp barrier
  noTileOmpPrefixSum(IC, IC, n);
  //ompPrefixSum(IC, IC, m);
#pragma omp single
  {
    nnzC = IC[n];
  }
}

inline int processCColI(QValue x[], bool* xb,
    const int iBnnz, const int iJB[], const QValue iB[],
        const int IA[], const int JA[], const QValue A[],
        int* iJC, QValue* iC) {
  int ip = 0;
  for(int jp = 0; jp < iBnnz; ++jp) {
    int j = iJB[jp];
    for(int tp = IA[j]; tp < IA[j + 1]; ++tp) {
      int t = JA[tp];
      if(xb[t] == false) {
        iJC[ip++] = t;
        xb[t] = true;
        x[t] = iB[jp] * A[tp];
      } else
        x[t] += iB[jp] * A[tp];
    }
  }
  for(int vp = 0; vp < ip; ++vp) {
    int v = iJC[vp];
    iC[vp] = x[v];
    x[v] = 0;
    xb[v] = false;
  }
  return ip;
}

