#ifndef MKL_CSR_KERNEL_H_
#define MKL_CSR_KERNEL_H_

#include "CSR.h"
#include "mkl_spblas.h"
#include "mkl.h"
#include "mkl_types.h"
#include "tools/ntimer.h"

void mkl_CSR_IC_nnzC(int IA[], int JA[],
          int IB[], int JB[],
          int m, int n,
        int* IC);
void mkl_CSR_SpMM(int IA[], int JA[], QValue A[],
    int IB[], int JB[], QValue B[],
    int* &IC, int* &JC, QValue* &C, int& nnzC,
    int m, int k, int n);
CSR mkl_spmm(CSR &A, CSR& B);
void mkl_CSR_RMCL_OneStep(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const int stride);
#endif
