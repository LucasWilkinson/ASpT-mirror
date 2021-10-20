#include "cpu_csr_kernel.h"

void rezero_xb(int i, const int IA[], const int JA[], const int IB[], const int JB[], int IC[], int iJC[], bool xb[]) {
  int ip = IC[i];
  int startp = ip;
  for (int vp = IA[i]; vp < IA[i + 1]; ++vp) {
    int v = JA[vp];
    for (int kp = IB[v]; kp < IB[v + 1]; ++kp) {
      int k = JB[kp];
      if (xb[k] == false) {
        iJC[ip - startp] = k;
        ++ip;
        xb[k] = true;
      }
    }
  }
  for (int jp = IC[i]; jp < ip; ++jp) {
    int j = iJC[jp - startp];
    xb[j] = false;
  }
  IC[i + 1] = ip;
}

void sequential_CSR_IC_nnzC(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n, bool xb[],
    int* IC, int& nnzC) {
  int* iJC = (int*)qcalloc(n + 1, sizeof(int), __FUNCTION__, __LINE__);
  IC[0] = 0;
  for (int i = 0; i < m; ++i) {
    rezero_xb(i, IA, JA, IB, JB, IC, iJC, xb);
  }
  free(iJC);
  nnzC = IC[m];
}
/*
 *
 *  commenting it to redefine this function for adding functionaliy of bin flops count
long long getSpMMFlops(const int IA[], const int JA[], const QValue A[], const int nnzA,
    const int IB[], const int JB[], const QValue B[], const int nnzB,
    const int m, const int k, const int n) {
  long long flops = 0;
  for (int i = 0; i < m; ++i) {
    long row_flops = 0;
    for (int jp = IA[i]; jp < IA[i + 1]; ++jp) {
      int j = JA[jp];
      long Brow_j_nnz = IB[j + 1] - IB[j];
      flops += Brow_j_nnz;
      row_flops += Brow_j_nnz;
    }
#ifdef profiling
    if(row_flops > 2) printf("rowid=%d : %ld row_nnz=%d\n", i, row_flops, IA[i + 1] - IA[i]);
#endif
  }
  return flops * 2;
} */

long long getSpMMFlops(const int IA[], const int JA[], const QValue A[], const int nnzA,
    const int IB[], const int JB[], const QValue B[], const int nnzB,
    const int m, const int k, const int n) {
  long long flops = 0;
  for (int i = 0; i < m; ++i) {
    long row_flops = 0;
    for (int jp = IA[i]; jp < IA[i + 1]; ++jp) {
      int j = JA[jp];
      long Brow_j_nnz = IB[j + 1] - IB[j];
      row_flops += Brow_j_nnz;
    }
    if(row_flops > 1024) flops+=row_flops;
  }
  return flops;
}



void sequential_CSR_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
    const int IB[], const int JB[], const QValue B[], const int nnzB,
    int* &IC, int* &JC, QValue* &C, int& nnzC,
    const int m, const int k, const int n) {
  IC = (int*)qcalloc(m + 1, sizeof(int), __FUNCTION__, __LINE__);
  bool* xb = (bool*)qcalloc(n, sizeof(bool), __FUNCTION__, __LINE__);
#ifdef profiling
  QValue now = time_in_mill_now();
#endif
  sequential_CSR_IC_nnzC(IA, JA, IB, JB, m, n, xb,
      IC, nnzC);
#ifdef profiling
  std::cout << "time passed seq nnzC " << time_in_mill_now() - now << std::endl;
#endif
  //printf("nnzC=%d\n",nnzC);
  JC = (int*)qmalloc(sizeof(int) * nnzC, __FUNCTION__, __LINE__);
  C = (QValue*)qmalloc(sizeof(QValue) * nnzC, __FUNCTION__, __LINE__);
  QValue* x = (QValue*)qcalloc(n, sizeof(QValue), __FUNCTION__, __LINE__);
  int ip = 0;
  for (int i = 0; i < m; ++i) {
    assert(ip == IC[i]);
    for (int jp = IA[i]; jp < IA[i + 1]; ++jp) {
      int j = JA[jp];
      for (int tp = IB[j]; tp < IB[j + 1]; ++tp) {
        int t = JB[tp];
        if (xb[t] == false) {
          JC[ip++] = t;
          xb[t] = true;
          x[t] = A[jp] * B[tp];
        }
        else
          x[t] += A[jp] * B[tp];
      }
    }
    for (int vp = IC[i]; vp < ip; ++vp) {
      int v = JC[vp];
      C[vp] = x[v];
      x[v] = 0;
      xb[v] = false;
    }
  }
  free(xb);
  free(x);
}
