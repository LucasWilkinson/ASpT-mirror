#include <omp.h>
#include <iostream>
#include "cpu_csc_kernel.h"
#include "tools/ntimer.h"
#include "tools/util.h"
#include "tools/prefixSum.h"
using namespace std;

long spmmCSCFootPrints(const int IB[], const int JB[],
    const int IA[], const int IC[],
    const int n,
    long *footPrintSum) {
  long footPrints = 0;
  footPrintSum[0] = 0;
  for (int i = 0; i < n; ++i) {
    long col_flops = 0;
    for (int jp = IB[i]; jp < IB[i + 1]; ++jp) {
      int j = JB[jp];
      long Acol_j_nnz = IA[j + 1] - IA[j];
      col_flops += Acol_j_nnz;
    }
    footPrints += col_flops + IC[i + 1] - IC[i] + 1;
    footPrintSum[i + 1] = footPrints;
  }
  return footPrints;
}


inline int footPrintsCColiCount(const int i, const int IA[], const int JA[], const int IB[], const int JB[], int iJC[], bool xb[], 
  int &footPrints) {
  if (IB[i] == IB[i + 1]) {
    return 0;
  }
  int count = -1;
  int vp = IB[i];
  int v = JB[vp];
  footPrints = 0;

  for (int kp = IA[v]; kp < IA[v+1]; ++kp) {
    
    int k = JA[kp];
    
    iJC[++count] = k;
    
    xb[k] = true;
  }
  footPrints += IA[v + 1] - IA[v];
  for (int vp = IB[i] + 1; vp < IB[i + 1]; ++vp) {
    
    int v = JB[vp];
    for (int kp = IA[v]; kp < IA[v+1]; ++kp) {
      int k = JA[kp];
      if(xb[k] == false) {
        iJC[++count] = k;
        xb[k] = true;
      }
    }
    footPrints += IA[v + 1] - IA[v];
  }
  ++count;
  for(int jp = 0; jp < count; ++jp) {
    int j = iJC[jp];
    xb[j] = false;
  }
  footPrints += count + 32 + (IB[i + 1] - IB[i]);
  //footPrints += count + 1;
  footPrints >>= 1; // The way to remove integer overflow in later prefix sum.
  
  return count;
}

/*
 * dynamic_omp_CSC_IC_nnzC_footprints reminder: this function must be called in #pragma omp parallel regions
 * to parallelly execution.
 * */
void dynamic_omp_CSC_IC_nnzC_footprints(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n, const thread_data_t& thread_data,
    int* IC, int& nnzC, int* footPrints, const int stride) {
  int *iJC = (int*)thread_data.index;
  bool *xb = thread_data.xb;

  memset(xb, 0, m);

#pragma omp for schedule(dynamic)
  for (int it = 0; it < n; it += stride) {
    int up = it + stride < n ? it + stride : n;
    for (int i = it; i < up; ++i) {
      IC[i] = footPrintsCColiCount(i, IA, JA, IB, JB, iJC, xb, footPrints[i]);
    }
  }
#pragma omp barrier
  noTileOmpPrefixSum(IC, IC, n);
  noTileOmpPrefixSum(footPrints, footPrints, n);
#pragma omp single
  {
    nnzC = IC[n];
  }
}


void static_omp_CSC_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride) {

  IC = (int*)malloc((n + 1) * sizeof(int));
  int* footPrints = (int*)malloc((n + 1) * sizeof(int));

  static int ends[MAX_THREADS_NUM];
#pragma omp parallel firstprivate(stride)
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();

    dynamic_omp_CSC_IC_nnzC_footprints(IA, JA, IB, JB, m, n, thread_datas[tid], IC, nnzC, footPrints, stride);

#pragma omp barrier
#pragma omp single
    {

      //spmmCSCFootPrints(IA, JA, IB, IC, n, footPrints);
      // VT commented below
      //printf("nthreads:%d\n", nthreads);
      arrayEqualPartition(footPrints, n, nthreads, ends);

    }
#pragma omp master
    {

      JC = (int*)malloc(sizeof(int) * nnzC);
      C = (QValue*)malloc(sizeof(QValue) * nnzC);

    }
    QValue *x = thread_datas[tid].x;
    int *index = thread_datas[tid].index;

    memset(index, -1, m * sizeof(int));

#pragma omp barrier

     int low = ends[tid];
     int high = ends[tid + 1];
     for (int i = low; i < high; ++i) {
    	
       indexProcessCColI(index,
           IB[i + 1] - IB[i], JB + IB[i], B + IB[i],
           IA, JA, A,
           JC + IC[i], C + IC[i]);
     }
    // #pragma omp for schedule(dynamic, stride)
    //   for(int i=0; i<n; i++) {
    //       indexProcessCColI(index,
    //         IB[i + 1] - IB[i], JB + IB[i], B + IB[i],
    //         IA, JA, A,
    //         JC + IC[i], C + IC[i]);
    //   }
    
  }
  free(footPrints);

}

void static_omp_CSC_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const int stride) {

    int nthreads = 8;
#pragma omp parallel
#pragma omp master
    nthreads = omp_get_num_threads();
    thread_data_t* thread_datas = allocateThreadDatas(nthreads, m);
    static_omp_CSC_SpMM(IA, JA, A, nnzA,
        IB, JB, B, nnzB,
        IC, JC, C, nnzC,
        m, k, n, thread_datas, stride);
    freeThreadDatas(thread_datas, nthreads);

}

