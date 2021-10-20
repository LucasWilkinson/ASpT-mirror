#include "mkl_csr_kernel.h"
#include "tools/util.h"
#include <omp.h>

void mkl_CSR_IC_nnzC(int IA[], int JA[],
          int IB[], int JB[],
          int m, int n,
        int* IC) {
  char trans = 'N';
  MKL_INT job = 1, sort = 7, nzmax = 0, ierr = -9;
  int k = m;
#ifdef FDOUBLE
  mkl_dcsrmultcsr(&trans, &job, &sort, &m, &n, &k, NULL, JA, IA,
      NULL, JB, IB, NULL, NULL, IC, NULL, &ierr);
#else
  mkl_scsrmultcsr(&trans, &job, &sort, &m, &n, &k, NULL, JA, IA,
      NULL, JB, IB, NULL, NULL, IC, NULL, &ierr);
#endif
}

void mkl_CSR_SpMM(int IA[], int JA[], QValue A[],
    int IB[], int JB[], QValue B[],
    int* &IC, int* &JC, QValue* &C, int& nnzC,
    int m, int k, int n) {
  char trans = 'N';
  assert(sizeof(int) == sizeof(MKL_INT));
  IC = (MKL_INT*)malloc((m + 1) * sizeof(MKL_INT));;
  double now = time_in_mill_now();
  mkl_CSR_IC_nnzC(IA, JA, IB, JB, m, n, IC);
  std::cout << "time passed mkl nnzC " << time_in_mill_now() - now << std::endl;
  nnzC = IC[m] - 1;
  MKL_INT sort = 7, ierr = -9;
  MKL_INT job = 2;
  C = (QValue*)malloc(nnzC * sizeof(QValue));
  JC = (MKL_INT*)malloc(nnzC * sizeof(MKL_INT));
#ifdef FDOUBLE
  mkl_dcsrmultcsr(&trans, &job, &sort, &m, &n, &k,
      A, JA, IA,
      B, JB, IB,
      C, JC, IC,
      &nnzC, &ierr);
#else
  mkl_scsrmultcsr(&trans, &job, &sort, &m, &n, &k,
      A, JA, IA,
      B, JB, IB,
      C, JC, IC,
      &nnzC, &ierr);
#endif
}

CSR mkl_spmm(CSR &A, CSR& B) {
  assert(A.cols == B.rows);
  int* IC;
  int* JC;
  QValue* C;
  int nnzC;
  mkl_CSR_SpMM(A.rowPtr, A.colInd, A.values,
      B.rowPtr, B.colInd, B.values,
      IC, JC, C, nnzC,
      A.rows, A.cols, B.cols);
  CSR csr(C, JC, IC, A.rows, B.cols, nnzC);
  return csr;
}

//Both matrix A, B and C are one based index.
void mkl_CSR_RMCL_OneStep(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const int stride) {
#ifdef profiling
  QValue now = time_in_mill_now();
#endif
  mkl_CSR_SpMM(const_cast<int*>(IA), const_cast<int*>(JA), const_cast<QValue*>(A),
      const_cast<int*>(IB), const_cast<int*>(JB), const_cast<QValue*>(B),
      IC, JC, C, nnzC,
      m, k, n);
#ifdef profiling
  printf("Time pass for mkl_CSR_SpMM ", time_in_mill_now() - now);
#endif
  //CSR tmpC(C, JC, IC, m, n, nnzC);
  //tmpC.toZeroBasedCSR();
  int* rowsNnz = (int*)malloc((m + 1) * sizeof(int));
#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
#pragma omp for schedule(dynamic, stride)
    for (int i = 0; i < m; ++i) {
        QValue *cQValues = C + IC[i] - 1; //-1 for one based IC index
        int *cColInd = JC + IC[i] - 1;
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
    int top = rowsNnz[0];
    for (int i = 1; i < m; ++i) {
      int up = IC[i] -1 + rowsNnz[i];
      const int preTop = top;
      for (int j = IC[i] - 1; j < up; ++j) {
        JC[top] = JC[j];
        C[top++] = C[j];
      }
      IC[i] = preTop + 1;
    }
    IC[m] = top + 1;
    free(rowsNnz);
    nnzC = top;
#ifdef profiling
    std::cout << "time passed without memory allocate " << time_in_mill_now() - now << std::endl;
#endif
  //tmpC.toOneBasedCSR();
  JC = (int*)realloc(JC, sizeof(int) * nnzC);
  C = (QValue*)realloc(C, sizeof(QValue) * nnzC);
#ifdef profiling
    std::cout << "time passed mkl_CSR_RMCL_OneStep " << time_in_mill_now() - now << std::endl;
#endif
}
