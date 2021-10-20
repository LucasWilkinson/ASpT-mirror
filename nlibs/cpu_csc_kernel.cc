#include "cpu_csc_kernel.h"

void rezero_xb_csc(int i, const int IA[], const int JA[], const int IB[], const int JB[], int IC[], int iJC[], bool xb[]) {
  int ip = IC[i];
  int startp = ip;
  for (int vp = IB[i]; vp < IB[i + 1]; ++vp) {
    int v = JB[vp];

    for (int kp = IA[v]; kp < IA[v + 1]; ++kp) {
      
      int k = JA[kp];
      
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

void sequential_CSC_IC_nnzC(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n, bool xb[],
    int* IC, int& nnzC) {
  int* iJC = (int*)qcalloc(m + 1, sizeof(int), __FUNCTION__, __LINE__);
  IC[0] = 0;
  for (int i = 0; i < n; ++i) {
    rezero_xb_csc(i, IA, JA, IB, JB, IC, iJC, xb);
  }
  free(iJC);
  nnzC = IC[n];
}


long long getCSCSpMMFlops(const int IA[], const int JA[], const QValue A[], const int nnzA,
    const int IB[], const int JB[], const QValue B[], const int nnzB,
    const int m, const int k, const int n) {
  long long flops = 0;
  for (int i = 0; i < n; ++i) {
    long col_flops = 0;
    for (int jp = IB[i]; jp < IB[i + 1]; ++jp) {
      int j = JB[jp];
      long Acol_j_nnz = IA[j + 1] - IA[j];
      col_flops += Acol_j_nnz;
    }
    if(col_flops > 1024) flops+=col_flops;
  }
  return flops;
}



void sequential_CSC_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
    const int IB[], const int JB[], const QValue B[], const int nnzB,
    int* &IC, int* &JC, QValue* &C, int& nnzC,
    const int m, const int k, const int n) {
  IC = (int*)qcalloc(n + 1, sizeof(int), __FUNCTION__, __LINE__);
  bool* xb = (bool*)qcalloc(m, sizeof(bool), __FUNCTION__, __LINE__);
#ifdef profiling
  QValue now = time_in_mill_now();
#endif
  sequential_CSC_IC_nnzC(IA, JA, IB, JB, m, n, xb,
      IC, nnzC);
#ifdef profiling
  std::cout << "time passed seq nnzC " << time_in_mill_now() - now << std::endl;
#endif
  //printf("nnzC=%d\n",nnzC);
  JC = (int*)qmalloc(sizeof(int) * nnzC, __FUNCTION__, __LINE__);
  C = (QValue*)qmalloc(sizeof(QValue) * nnzC, __FUNCTION__, __LINE__);
  QValue* x = (QValue*)qcalloc(m, sizeof(QValue), __FUNCTION__, __LINE__);
  int ip = 0;
  for (int i = 0; i < n; ++i) {
    assert(ip == IC[i]);
    for (int jp = IB[i]; jp < IB[i + 1]; ++jp) {
      int j = JB[jp];
      for (int tp = IA[j]; tp < IA[j + 1]; ++tp) {
        int t = JA[tp];
        if (xb[t] == false) {
          JC[ip++] = t;
          xb[t] = true;
          x[t] = B[jp] * A[tp];
        }
        else
          x[t] += B[jp] * A[tp];
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
