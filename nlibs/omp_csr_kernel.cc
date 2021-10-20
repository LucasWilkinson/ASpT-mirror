#include <omp.h>
#include <iostream>
#include "tools/ntimer.h"
#include "cpu_csr_kernel.h"
#include "tools/util.h"
#include "tools/prefixSum.h"
using namespace std;

#ifdef USE_MIC
#define AGGR
#endif

thread_data_t* allocateThreadDatas(int nthreads, int n) {
#ifdef profiling
  QValue now = time_in_mill_now();
#endif
  thread_data_t* thread_datas = (thread_data_t*)malloc(nthreads * sizeof(thread_data_t));
#ifdef AGGR
  QValue *xs = (QValue*)qmalloc((n * sizeof(QValue) + LEVEL1_DCACHE_LINESIZE) * nthreads, __FUNCTION__, __LINE__);
  bool *xbs = (bool*)qmalloc((n + LEVEL1_DCACHE_LINESIZE) * nthreads, __FUNCTION__, __LINE__);
  //bool *xbs = (bool*)qcalloc((n + LEVEL1_DCACHE_LINESIZE) * nthreads, sizeof(bool), __FUNCTION__, __LINE__);
  int *indices = (int*)qmalloc((n * sizeof(int) + LEVEL1_DCACHE_LINESIZE) * nthreads, __FUNCTION__, __LINE__);
  //memset(indices, -1, (n * sizeof(int) + LEVEL1_DCACHE_LINESIZE) * nthreads);
#endif
  for(int i = 0; i < nthreads; i++) {
#ifndef AGGR
    thread_datas[i].init(n);
#else
    QValue *x = (QValue*)((char*)xs + i * (n * sizeof(QValue) + LEVEL1_DCACHE_LINESIZE));
    //bool *xb = (bool*)((char*)xbs + i * (n + LEVEL1_DCACHE_LINESIZE));
    bool *xb = (bool*)x;
    int *index = (int*)((char*)indices + i * (n * sizeof(int) + LEVEL1_DCACHE_LINESIZE));
    thread_datas[i].init(x, xb, index);
#endif
  }
#ifdef profiling
  printf("Time passed for %s in %lf milliseconds\n", __func__, time_in_mill_now() - now);
#endif
  return thread_datas;
}

void freeThreadDatas(thread_data_t* thread_datas, int nthreads) {
#ifdef profiling
  QValue now = time_in_mill_now();
#endif
#ifndef AGGR
  for(int i = 0; i < nthreads; i++) {
    thread_datas[i].~thread_data_t();
  }
#else
  free(thread_datas[0].x);
  //free(thread_datas[0].xb);
  free(thread_datas[0].index);
  free(thread_datas);
#endif
#ifdef profiling
  printf("Time passed for %s in %lf milliseconds\n", __func__, time_in_mill_now() - now);
#endif
}

thread_data_t* allocateThreadDatas_memory_Managed(int nthreads, int n) {

  thread_data_t* thread_datas = (thread_data_t*)malloc(nthreads * sizeof(thread_data_t));
  for(int i = 0; i < nthreads; i++) {
    thread_datas[i].init_memory_Managed(n);
  }
  return thread_datas;
}

void freeThreadDatas_memory_Managed(thread_data_t* thread_datas, int nthreads) {
  for(int i = 0; i < nthreads; i++) {
    free(thread_datas[i].xbp);
    thread_datas[i].xbp=NULL;
  }
}

void omp_CSR_IC_nnzC_Wrapper(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n, const thread_data_t thread_datas[],
    int* IC, int& nnzC, const int stride) {
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    omp_CSR_IC_nnzC(IA, JA, IB, JB,
    m, n, thread_datas[tid],
    IC, nnzC, stride);
  }
}

/*
 * static_omp_CSR_IC_nnzC reminder: this function must be called in #pragma omp parallel regions
 * to parallelly execution.
 * */
void static_omp_CSR_IC_nnzC(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n, const thread_data_t& thread_data,
    int* IC, int& nnzC, const int stride, const int ends[], const int tid) {
  int *iJC = (int*)thread_data.index;
  bool *xb = thread_data.xb;
  memset(xb, 0, n);
#pragma omp barrier
  int low = ends[tid];
  int high = ends[tid + 1];
  for (int i = low; i < high; ++i) {
    IC[i] = cRowiCount(i, IA, JA, IB, JB, iJC, xb);
  }
#pragma omp barrier
  noTileOmpPrefixSum(IC, IC, m);
#pragma omp single
  {
    nnzC = IC[m];
  }
}

/*
 * omp_CSR_IC_nnzC reminder: this function must be called in #pragma omp parallel regions
 * to parallelly execution.
 * */
void omp_CSR_IC_nnzC(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n, const thread_data_t& thread_data,
    int* IC, int& nnzC, const int stride) {
  int *iJC = (int*)thread_data.index;
  bool *xb = thread_data.xb;
  memset(xb, 0, n);
#pragma omp for schedule(dynamic)
  for (int it = 0; it < m; it += stride) {
    int up = it + stride < m ? it + stride : m;
    for (int i = it; i < up; ++i) {
      IC[i] = cRowiCount(i, IA, JA, IB, JB, iJC, xb);
    }
  }
#pragma omp barrier
  noTileOmpPrefixSum(IC, IC, m);
  //ompPrefixSum(IC, IC, m);
#pragma omp single
  {
    nnzC = IC[m];
  }
}

inline int processCRowI(QValue x[], bool* xb,
    const int iAnnz, const int iJA[], const QValue iA[],
        const int IB[], const int JB[], const QValue B[],
        int* iJC, QValue* iC) {
  int ip = 0;
  for(int jp = 0; jp < iAnnz; ++jp) {
    int j = iJA[jp];
    for(int tp = IB[j]; tp < IB[j + 1]; ++tp) {
      int t = JB[tp];
      if(xb[t] == false) {
        iJC[ip++] = t;
        xb[t] = true;
        x[t] = iA[jp] * B[tp];
      } else
        x[t] += iA[jp] * B[tp];
    }
    //x[JB[IBj : IBj1 - IBj]] += iA[jp] * B[IBj : IBj1 - IBj];
    //xb[JB[IBj : IBj1 - IBj]] = true;
  }
  for(int vp = 0; vp < ip; ++vp) {
    int v = iJC[vp];
    iC[vp] = x[v];
    x[v] = 0;
    xb[v] = false;
  }
  return ip;
}

void omp_CSR_RMCL_OneStep(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride) {
    IC = (int*)calloc(m + 1, sizeof(int));
    int* rowsNnz = (int*)malloc((m + 1) * sizeof(int));
#pragma omp parallel firstprivate(stride) //num_threads(1)
    {
      int tid = omp_get_thread_num();
      omp_CSR_IC_nnzC(IA, JA, IB, JB, m, n, thread_datas[tid], IC, nnzC, stride);
#pragma omp master
      {
        JC = (int*)malloc(sizeof(int) * nnzC);
        C = (QValue*)malloc(sizeof(QValue) * nnzC);
      }
      QValue *x = thread_datas[tid].x;
      bool *xb = thread_datas[tid].xb;
      int *index = thread_datas[tid].index;
      memset(index, -1, n * sizeof(int));
#pragma omp barrier
#pragma omp for schedule(dynamic)
      for (int it = 0; it < m; it += stride) {
        int up = it + stride < m ? it + stride : m;
        for (int i = it; i < up; ++i) {
          QValue *cQValues = C + IC[i];
          int *cColInd = JC + IC[i];
          //processCRowI(x, xb,
          indexProcessCRowI(index,
              IA[i + 1] - IA[i], JA + IA[i], A + IA[i],
              IB, JB, B,
              cColInd, cQValues);
          int count = IC[i + 1] - IC[i];
          arrayInflationR2(cQValues, count, cQValues);
          pair<QValue, QValue> maxSum = arrayMaxSum(cQValues, count);
          QValue rmax = maxSum.first, rsum = maxSum.second;
          QValue thresh = computeThreshold(rsum / count, rmax);
          arrayThreshPruneNormalize(thresh, cColInd, cQValues,
              &count, cColInd, cQValues);
          rowsNnz[i] = count;
        }
      }
    }
    matrix_relocation(rowsNnz, m, IC, JC, C, nnzC);
    free(rowsNnz);
}

//This function must be called in OpenMP parallel region
void omp_matrix_relocation(int rowsNnz[], const int m, const int tid, const int stride,
        int* &IC, int* &JC, QValue* &C, int& nnzC) {
#ifdef profiling
    QValue rnow = time_in_mill_now();
#endif
 int *oJC = JC;
 QValue *oC = C;
 noTileOmpPrefixSum(rowsNnz, rowsNnz, m);
#pragma omp master
  {
    nnzC = rowsNnz[m];
    JC = (int*)qmalloc(sizeof(int) * nnzC, __FUNCTION__, __LINE__);
    C = (QValue*)qmalloc(sizeof(QValue) * nnzC, __FUNCTION__, __LINE__);
  }
#pragma omp barrier
#pragma omp for schedule(dynamic, stride)
  for (int i = 0; i < m; ++i) {
    int up = IC[i] + rowsNnz[i + 1] - rowsNnz[i];
    int top = rowsNnz[i] - 1;
    for (int j = IC[i]; j < up; ++j) {
      JC[++top] = oJC[j];
      C[top] = oC[j];
    }
    IC[i] = rowsNnz[i];
  }
#pragma omp barrier
#pragma omp master
  {
    IC[m] = rowsNnz[m];
    free(oJC);
    free(oC);
  }
#ifdef profiling
    printf("time passed for omp relocate IC, JC and C %lf on thread %d\n", time_in_mill_now() - rnow, tid);
#endif
}

void omp_CSR_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride) {
    IC = (int*)malloc((m + 1) * sizeof(int));
#ifdef profiling
    QValue now = time_in_mill_now();
#endif
#pragma omp parallel firstprivate(stride)
    {
      int tid = omp_get_thread_num();
#ifdef profiling
#pragma omp master
      {
        now = time_in_mill_now();
      }
#endif
      omp_CSR_IC_nnzC(IA, JA, IB, JB, m, n, thread_datas[tid], IC, nnzC, stride);
#pragma omp master
      {
#ifdef profiling
        std::cout << "time passed omp nnzC " << time_in_mill_now() - now << std::endl;
#endif
        JC = (int*)malloc(sizeof(int) * nnzC);
        C = (QValue*)malloc(sizeof(QValue) * nnzC);
      }
      QValue *x = thread_datas[tid].x;
      //bool *xb = thread_datas[tid].xb;
      int *index = thread_datas[tid].index;
      memset(index, -1, n * sizeof(int));
#pragma omp barrier
#pragma omp for schedule(dynamic) nowait
      for (int it = 0; it < m; it += stride) {
        int up = it + stride < m ? it + stride : m;
        for (int i = it; i < up; ++i) {
          //processCRowI(x, xb,
          indexProcessCRowI(index,
              IA[i + 1] - IA[i], JA + IA[i], A + IA[i],
              IB, JB, B,
              JC + IC[i], C + IC[i]);
        }
      }
    }
// #pragma omp for schedule(static, ((m + nthreads - 1) / nthreads)) nowait
//         for (int i = 0; i < m; ++i) {
//           //processCRowI(x, xb,
//           indexProcessCRowI(index,
//               IA[i + 1] - IA[i], JA + IA[i], A + IA[i],
//               IB, JB, B,
//               JC + IC[i], C + IC[i]);
//         }
//      }
#ifdef profiling
    std::cout << "time passed without memory allocate" << time_in_mill_now() - now << std::endl;
#endif
}

void omp_CSR_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const int stride) {
#ifdef profiling
    QValue now = time_in_mill_now();
#endif
    int nthreads = 8;
#pragma omp parallel
#pragma omp master
    nthreads = omp_get_num_threads();
    thread_data_t* thread_datas = allocateThreadDatas(nthreads, n);
    omp_CSR_SpMM(IA, JA, A, nnzA,
        IB, JB, B, nnzB,
        IC, JC, C, nnzC,
        m, k, n, thread_datas, stride);
    freeThreadDatas(thread_datas, nthreads);
#ifdef profiling
    std::cout << "time passed for omp_CSR_SpMM total " <<  time_in_mill_now() - now << std::endl;
#endif
}

void noindex_somp_CSR_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride) {
  IC = (int*)malloc((m + 1) * sizeof(int));
  int* footPrints = (int*)malloc((m + 1) * sizeof(int));
  static int ends[MAX_THREADS_NUM];
#pragma omp parallel firstprivate(stride)
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    dynamic_omp_CSR_IC_nnzC_footprints(IA, JA, IB, JB, m, n, thread_datas[tid], IC, nnzC, footPrints, stride);
#pragma omp barrier
#pragma omp single
    {
      arrayEqualPartition(footPrints, m, nthreads, ends);
    }
#pragma omp master
    {
      JC = (int*)malloc(sizeof(int) * nnzC);
      C = (QValue*)malloc(sizeof(QValue) * nnzC);
    }
    QValue *x = thread_datas[tid].x;
    bool *xb = (bool*)thread_datas[tid].index;
    memset(xb, 0, n * sizeof(bool));
#pragma omp barrier
    int low = ends[tid];
    int high = ends[tid + 1];
    for (int i = low; i < high; ++i) {
      processCRowI(x, xb,
          IA[i + 1] - IA[i], JA + IA[i], A + IA[i],
          IB, JB, B,
          JC + IC[i], C + IC[i]);
    }
  }
  free(footPrints);
}

void noindex_somp_CSR_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const int stride) {
#ifdef profiling
    QValue now = time_in_mill_now();
#endif
    int nthreads = 8;
#pragma omp parallel
#pragma omp master
    nthreads = omp_get_num_threads();
    thread_data_t* thread_datas = allocateThreadDatas(nthreads, n);
    noindex_somp_CSR_SpMM(IA, JA, A, nnzA,
        IB, JB, B, nnzB,
        IC, JC, C, nnzC,
        m, k, n, thread_datas, stride);
    freeThreadDatas(thread_datas, nthreads);
#ifdef profiling
    std::cout << "time passed for static_omp_CSR_SpMM total " << time_in_mill_now() - now << std::endl;
#endif
}
