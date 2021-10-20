#include <omp.h>
#include <iostream>
#include "cpu_csr_kernel.h"
#include "tools/ntimer.h"
#include "tools/util.h"
#include "tools/prefixSum.h"

using namespace std;

void group_CSR_flops(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n,
    int* IC, int& nnzC, int* rowFlops, int* groups, int *tops, const int stride) {
  int c0 = 0, c1 = 0, c2 = 0, c4 = 0, c8 = 0, c16 = 0, c16p = 0;
#pragma omp parallel for reduction(+:c0, c1, c2, c4, c8, c16, c16p)\
  schedule(dynamic, stride)
  for (int i = 0; i < m; ++i) {
    int tmpRowFlops = 0;
    for (int jp = IA[i]; jp < IA[i + 1]; ++jp) {
      int j = JA[jp];
      int BrowFlops = IB[j + 1] - IB[j];
      tmpRowFlops += BrowFlops;
    }
    if (tmpRowFlops <= 0) ++c0;
    else if (tmpRowFlops <= 1) ++c1;
    else if (tmpRowFlops <= 2) ++c2;
    else if (tmpRowFlops <= 4) ++c4;
    else if (tmpRowFlops <= 8) ++c8;
    else if (tmpRowFlops <= 16) ++c16;
    else ++c16p;
    rowFlops[i] = tmpRowFlops;
  }
  assert (c0 + c1 + c2 + c4 + c8 + c16 + c16p == m);
  int top0 = 0, top1 = c0, top2 = c0 + c1, top4 = c0 + c1 + c2;
  int top8 = c0 + c1 + c2 + c4, top16 = c0 + c1 + c2 + c4 + c8, top16p = c0 + c1 + c2 + c4 + c8 + c16;
  tops[0] = top0; tops[1] = top1; tops[2] = top2;
  tops[3] = top4; tops[4] = top8; tops[5] = top16; tops[6] = top16p; tops[7] = top16p + c16p;
  for (int i = 0; i < m; ++i) {
    if (rowFlops[i] <= 0) {
      groups[top0++] = i;
      IC[i] = 0;
    } else if (rowFlops[i] <= 1) {
      groups[top1++] = i;
      IC[i] = 1;
    } else if (rowFlops[i] <= 2) groups[top2++] = i;
    else if (rowFlops[i] <= 4) groups[top4++] = i;
    else if (rowFlops[i] <= 8) groups[top8++] = i;
    else if (rowFlops[i] <= 16) groups[top16++] = i;
    else groups[top16p++] = i;
  }
  assert (top16p == tops[7]);
}

void group_CSR_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride) {
  IC = (int*)malloc((m + 1) * sizeof(int));
  int* flops = (int*)malloc((m + 1) * sizeof(int));
  int* groups = (int*)malloc((m + 1) * sizeof(int));
  static int groupPtr[8];
  static int ends[MAX_THREADS_NUM];
  group_CSR_flops(IA, JA, IB, JB, m, n, IC, nnzC, flops, groups, groupPtr, stride);
#pragma omp parallel firstprivate(stride)
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    int *iJC = (int*)thread_datas[tid].index;
    bool *xb = thread_datas[tid].xb;
    memset(xb, 0, n);
    for (int gid = 2; gid < 7; ++gid) {
      int groupSize = groupPtr[gid + 1] - groupPtr[gid];
      int low = groupPtr[gid] + ((long long) groupSize) * tid / nthreads;
      int high = groupPtr[gid] + ((long long) groupSize) * (tid + 1) / nthreads;
      for (int i = low; i < high; ++i) {
        int r = groups[i];
        IC[r] = cRowiCount(r, IA, JA, IB, JB, iJC, xb);
      }
    }
#pragma omp barrier
    noTileOmpPrefixSum(IC, IC, m);
#pragma omp master
    {
      nnzC = IC[m];
      JC = (int*)malloc(sizeof(int) * nnzC);
      C = (QValue*)malloc(sizeof(QValue) * nnzC);
    }
    QValue *x = thread_datas[tid].x;
    int *index = thread_datas[tid].index;
    memset(index, -1, n * sizeof(int));
#pragma omp barrier
    for (int gid = 1; gid < 7; ++gid) {
      int groupSize = groupPtr[gid + 1] - groupPtr[gid];
      int low = groupPtr[gid] + ((long long) groupSize) * tid / nthreads;
      int high = groupPtr[gid] + ((long long) groupSize) * (tid + 1) / nthreads;
      for (int i = low; i < high; ++i) {
        int r = groups[i];
        indexProcessCRowI(index,
            IA[r + 1] - IA[r], JA + IA[r], A + IA[r],
            IB, JB, B,
            JC + IC[r], C + IC[r]);
      }
    }
  }
  free(flops);
}

void group_CSR_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
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
    group_CSR_SpMM(IA, JA, A, nnzA,
        IB, JB, B, nnzB,
        IC, JC, C, nnzC,
        m, k, n, thread_datas, stride);
    freeThreadDatas(thread_datas, nthreads);
#ifdef profiling
    std::cout << "time passed for group_CSR_SpMM total " <<  time_in_mill_now() - now << std::endl;
#endif
}
