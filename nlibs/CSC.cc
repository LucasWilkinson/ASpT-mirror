/*
 * CSC.cpp
 *
 *  Created on: Aug 8, 2017
 *      Author: vineeth
 */
#include "CSC.h"
#include "tools/util.h"
#include "tools/stats.h"
#include "tools/qmalloc.h"
#include <vector>
#include <algorithm>
#include <omp.h>
#ifdef enable_gpu
#include "gpus/cuda_handle_error.h"
#endif
// #include "gpus/gpu_CSC_kernel.h"
// #ifdef enable_mkl
// #include "mkls/mkl_csc_kernel.h"
// #endif
// #ifdef enable_cilk
// #include "cilks/cilk_csc_kernel.h"
// #endif

// void CSC::matrixColReorder(const int* ranks) const {
//   int* ncolPtr = (int*)qmalloc((cols + 1) * sizeof(int), __FUNCTION__, __LINE__);
//   int* nrowInd= (int*)qmalloc(nnz * sizeof(int), __FUNCTION__, __LINE__);
//   QValue* nvalues = (QValue*)qmalloc(nnz * sizeof(QValue), __FUNCTION__, __LINE__);
//   ncolPtr[0] = 0;
//   for (int i = 0; i < cols; ++i) {
//     int count = colPtr[ranks[i] + 1] - colPtr[ranks[i]];
//     ncolPtr[i + 1] = ncolPtr[i] + count;
//     memcpy(nrowInd + ncolPtr[i], rowInd + colPtr[ranks[i]],
//         count * sizeof(int));
//     memcpy(nvalues + ncolPtr[i], values + colPtr[ranks[i]],
//         count * sizeof(QValue));
//   }
//   memcpy(colPtr, ncolPtr, (cols + 1) * sizeof(int));
//   memcpy(rowInd, nrowInd, nnz * sizeof(int));
//   memcpy(values, nvalues, nnz * sizeof(QValue));
//   free(ncolPtr);
//   free(nrowInd);
//   free(nvalues);
// }

long CSC::spmmFlops(const CSC& B) const {
  long flops = getSpMMFlops(this->colPtr, this->rowInd, this->values, this->nnz,
      B.colPtr, B.rowInd, B.values, B.nnz, rows, cols, B.cols);
  return flops;
}

std::vector<int> CSC::multiFlopsStats(const CSC& B) const {
  std::vector<int> stats = flopsStats(this->colPtr, this->rowInd, this->values, this->nnz,
      B.colPtr, B.rowInd, B.values, B.nnz, rows, cols, B.cols);
  return stats;
}


CSC CSC::spmm(const CSC& B) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  QValue* C;
  int nnzC;
  sequential_CSC_SpMM(this->colPtr, this->rowInd, this->values, this->nnz,
      B.colPtr, B.rowInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols);
  CSC CSC(C, JC, IC, this->rows, B.cols, nnzC);
  return CSC;
}

void CSC::makeOrdered() {
  for (int i = 0; i < cols; ++i) {
    std::vector<std::pair<int, QValue> > colv;
    for (int jp = colPtr[i]; jp < colPtr[i + 1]; ++jp) {
      colv.push_back(std::make_pair(rowInd[jp], values[jp]));
    }
    std::sort(colv.begin(), colv.end());
    int iter = 0;
    for (int jp = colPtr[i]; jp < colPtr[i + 1]; ++jp, ++iter) {
      rowInd[jp] = colv[iter].first;
      values[jp] = colv[iter].second;
    }
  }
}

void CSC::averAndNormRowQValue() {
  for (int i = 0; i < cols; ++i) {
    int count = colPtr[i + 1] - colPtr[i];
    for (int j = colPtr[i]; j < colPtr[i + 1]; ++j) {
      values[j] = 1.0 / count;
    }
  }
}

CSC CSC::deepCopy() {
  int* bcolPtr = (int*)qmalloc((cols + 1) * sizeof(int), __FUNCTION__, __LINE__);
	QValue* bvalues = (QValue*)qmalloc(nnz * sizeof(QValue), __FUNCTION__, __LINE__);
  int* browInd = (int*)qmalloc(nnz * sizeof(int), __FUNCTION__, __LINE__);
  memcpy(bcolPtr, colPtr, (cols + 1) * sizeof(int));
  memcpy(bvalues, values, nnz * sizeof(QValue));
  memcpy(browInd, rowInd, nnz * sizeof(int));
  CSC B(bvalues, browInd, bcolPtr, rows, cols, nnz);
  return B;
}

CSC CSC::somp_spmm(const CSC& B, const int stride) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  QValue* C;
  int nnzC;
  static_omp_CSC_SpMM(this->colPtr, this->rowInd, this->values, this->nnz,
      B.colPtr, B.rowInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols, stride);
  CSC CSC(C, JC, IC, this->rows, B.cols, nnzC);
  return CSC;
}

// CSC CSC::somp_spmm_memory_Managed(const CSC& B, const int stride) const {
//   assert(this->cols == B.rows);
//   int* IC;
//   int* JC;
//   QValue* C;
//   int nnzC;
//   static_omp_CSC_SpMM_memory_Managed(this->colPtr, this->rowInd, this->values, this->nnz,
//       B.colPtr, B.rowInd, B.values, B.nnz,
//       IC, JC, C, nnzC,
//       this->rows, this->cols, B.cols, stride);
//   CSC CSC(C, JC, IC, this->rows, B.cols, nnzC);
//   return CSC;
// }



CSC CSC::somp_spmm(thread_data_t* thread_datas, const CSC& B, const int stride) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  QValue* C;
  int nnzC;
  int nthreads = 8;
#pragma omp parallel
#pragma omp master
  nthreads = omp_get_num_threads();
  static_omp_CSC_SpMM(this->colPtr, this->rowInd, this->values, this->nnz,
      B.colPtr, B.rowInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols, thread_datas, stride);
  CSC CSC(C, JC, IC, this->rows, B.cols, nnzC);
  return CSC;
}



// CSC CSC::flops_spmm(const CSC& B, const int stride) const {
//   assert(this->cols == B.rows);
//   int* IC;
//   int* JC;
//   QValue* C;
//   int nnzC;
//   flops_omp_CSC_SpMM(this->colPtr, this->rowInd, this->values, this->nnz,
//       B.colPtr, B.rowInd, B.values, B.nnz,
//       IC, JC, C, nnzC,
//       this->rows, this->cols, B.cols, stride);
//   CSC CSC(C, JC, IC, this->rows, B.cols, nnzC);
//   return CSC;
// }

// CSC CSC::group_spmm(const CSC& B, const int stride) const {
//   assert(this->cols == B.rows);
//   int* IC;
//   int* JC;
//   QValue* C;
//   int nnzC;
//   group_CSC_SpMM(this->colPtr, this->rowInd, this->values, this->nnz,
//       B.colPtr, B.rowInd, B.values, B.nnz,
//       IC, JC, C, nnzC,
//       this->rows, this->cols, B.cols, stride);
//   CSC CSC(C, JC, IC, this->rows, B.cols, nnzC);
//   return CSC;
// }


/* This method returns the norm of A-B. Remember, it assumes
 * that the adjacency lists in both A and B are sorted in
 * ascending order. */
QValue CSC::differs(const CSC& B) const {
  QValue sum = 0;
  int i, j, k;
  for (i = 0; i < cols; ++i) {
    for (j = colPtr[i], k = B.colPtr[i];
        j < colPtr[i + 1] && k < B.colPtr[i + 1];) {
      QValue a = values[j];
      QValue b = B.values[k];
      if (rowInd[j] == rowInd[k]) {
        sum += (a - b) * (a - b);
        ++j, ++k;
      } else if (rowInd[j] < rowInd[k]){
        sum += a * a;
        ++j;
      } else {
        sum += b * b;
        ++k;
      }
    }
    for (; j < colPtr[i + 1]; ++j) {
      sum += values[j] * values[j];
    }
    for (; k < colPtr[i + 1]; ++k) {
      sum += B.values[k] * B.values[k];
    }
  }
  return sum;
}

vector<int> CSC::nnzStats() const {
  std::vector<int> stats(30, 0);
  for (int i = 0; i < cols; ++i) {
    long stat = colPtr[i + 1] - colPtr[i];
    pushToStats(colPtr[i + 1] - colPtr[i], stats);
  }
  return stats;
}


long long CSC::spMMFlops(const CSC &B) const {
  return getSpMMFlops(this->colPtr, this->rowInd, this->values, this->nnz,
      B.colPtr, B.rowInd, B.values, B.nnz,
      this->rows, this->cols, B.cols);
}

// void CSC::outputSpMMStats(const CSC &B) const {
//   long long flops = this->spMMFlops(B);
//   CSC C = this->omp_spmm(B, 512);
//   int cNnz = C.nnz;
//   C.dispose();
//   printf("flops=%lld\tcNnz=%d\trows=%d\tflops/rows=%lf cnnz/rows=%lf\n", flops, cNnz, rows, (QValue)(flops) / rows, (QValue)(cNnz) / rows);
// }


