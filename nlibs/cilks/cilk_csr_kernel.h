#ifndef CILK_CSR_KERNEL_H_
#define CILK_CSR_KERNEL_H_

#include "cpu_csr_kernel.h"
#include "tools/ntimer.h"
#include "tools/util.h"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
//#include <iostream>
#include <omp.h>
//using namespace std;

void cilk_CSR_RMCL_OneStep(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride);
#endif
