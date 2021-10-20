/*
 * CSR.cpp
 *
 *  Created on: Oct 3, 2013
 *      Author: niuq
 */
#include "CSR.h"
#include "tools/util.h"
#include "tools/stats.h"
#include "tools/qmalloc.h"
#ifdef enable_mkl
#include "mkls/mkl_csr_kernel.h"
#endif
#include <vector>
#include <algorithm>
#include <omp.h>
#ifdef enable_gpu
#include "gpus/cuda_handle_error.h"
#endif
//#include "gpus/gpu_csr_kernel.h"
#ifdef enable_cilk
#include "cilks/cilk_csr_kernel.h"
#endif

void CSR::matrixRowReorder(const int* ranks) const {
  int* nrowPtr = (int*)qmalloc((rows + 1) * sizeof(int), __FUNCTION__, __LINE__);
  int* ncolInd= (int*)qmalloc(nnz * sizeof(int), __FUNCTION__, __LINE__);
  QValue* nvalues = (QValue*)qmalloc(nnz * sizeof(QValue), __FUNCTION__, __LINE__);
  nrowPtr[0] = 0;
  for (int i = 0; i < rows; ++i) {
    int count = rowPtr[ranks[i] + 1] - rowPtr[ranks[i]];
    nrowPtr[i + 1] = nrowPtr[i] + count;
    memcpy(ncolInd + nrowPtr[i], colInd + rowPtr[ranks[i]],
        count * sizeof(int));
    memcpy(nvalues + nrowPtr[i], values + rowPtr[ranks[i]],
        count * sizeof(QValue));
  }
  memcpy(rowPtr, nrowPtr, (rows + 1) * sizeof(int));
  memcpy(colInd, ncolInd, nnz * sizeof(int));
  memcpy(values, nvalues, nnz * sizeof(QValue));
  free(nrowPtr);
  free(ncolInd);
  free(nvalues);
}

long CSR::spmmFlops(const CSR& B) const {
  long flops = getSpMMFlops(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz, rows, cols, B.cols);
  return flops;
}

std::vector<int> CSR::multiFlopsStats(const CSR& B) const {
  std::vector<int> stats = flopsStats(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz, rows, cols, B.cols);
  return stats;
}


CSR CSR::spmm(const CSR& B) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  QValue* C;
  int nnzC;
  sequential_CSR_SpMM(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols);
  CSR csr(C, JC, IC, this->rows, B.cols, nnzC);
  return csr;
}

void CSR::makeOrdered() {
  for (int i = 0; i < rows; ++i) {
    std::vector<std::pair<int, QValue> > rowv;
    for (int jp = rowPtr[i]; jp < rowPtr[i + 1]; ++jp) {
      rowv.push_back(std::make_pair(colInd[jp], values[jp]));
    }
    std::sort(rowv.begin(), rowv.end());
    int iter = 0;
    for (int jp = rowPtr[i]; jp < rowPtr[i + 1]; ++jp, ++iter) {
      colInd[jp] = rowv[iter].first;
      values[jp] = rowv[iter].second;
    }
  }
}

void CSR::averAndNormRowQValue() {
  for (int i = 0; i < rows; ++i) {
    int count = rowPtr[i + 1] - rowPtr[i];
    for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j) {
      values[j] = 1.0 / count;
    }
  }
}

CSR CSR::deepCopy() {
  int* browPtr = (int*)qmalloc((rows + 1) * sizeof(int), __FUNCTION__, __LINE__);
	QValue* bvalues = (QValue*)qmalloc(nnz * sizeof(QValue), __FUNCTION__, __LINE__);
  int* bcolInd = (int*)qmalloc(nnz * sizeof(int), __FUNCTION__, __LINE__);
  memcpy(browPtr, rowPtr, (rows + 1) * sizeof(int));
  memcpy(bvalues, values, nnz * sizeof(QValue));
  memcpy(bcolInd, colInd, nnz * sizeof(int));
  CSR B(bvalues, bcolInd, browPtr, rows, cols, nnz);
  return B;
}

CSR CSR::somp_spmm(const CSR& B, const int stride) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  QValue* C;
  int nnzC;
  static_omp_CSR_SpMM(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols, stride);
  CSR csr(C, JC, IC, this->rows, B.cols, nnzC);
  return csr;
}

CSR CSR::somp_spmm_memory_Managed(const CSR& B, const int stride) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  QValue* C;
  int nnzC;
  static_omp_CSR_SpMM_memory_Managed(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols, stride);
  CSR csr(C, JC, IC, this->rows, B.cols, nnzC);
  return csr;
}

CSR CSR::omp_spmm(const CSR& B, const int stride) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  QValue* C;
  int nnzC;
  omp_CSR_SpMM(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols, stride);
  CSR csr(C, JC, IC, this->rows, B.cols, nnzC);
  return csr;
}

CSR CSR::somp_spmm(thread_data_t* thread_datas, const CSR& B, const int stride) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  QValue* C;
  int nnzC;
  int nthreads = 8;
#pragma omp parallel
#pragma omp master
  nthreads = omp_get_num_threads();
  static_omp_CSR_SpMM(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols, thread_datas, stride);
  CSR csr(C, JC, IC, this->rows, B.cols, nnzC);
  return csr;
}

CSR CSR::omp_spmm(thread_data_t* thread_datas, const CSR& B, const int stride) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  QValue* C;
  int nnzC;
  omp_CSR_SpMM(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols, thread_datas, stride);
  CSR csr(C, JC, IC, this->rows, B.cols, nnzC);
  return csr;
}

CSR CSR::noindex_somp_spmm(const CSR& B, const int stride) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  QValue* C;
  int nnzC;
  noindex_somp_CSR_SpMM(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols, stride);
  CSR csr(C, JC, IC, this->rows, B.cols, nnzC);
  return csr;
}

CSR CSR::flops_spmm(const CSR& B, const int stride) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  QValue* C;
  int nnzC;
  flops_omp_CSR_SpMM(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols, stride);
  CSR csr(C, JC, IC, this->rows, B.cols, nnzC);
  return csr;
}

CSR CSR::group_spmm(const CSR& B, const int stride) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  QValue* C;
  int nnzC;
  group_CSR_SpMM(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols, stride);
  CSR csr(C, JC, IC, this->rows, B.cols, nnzC);
  return csr;
}

/* This method returns the norm of A-B. Remember, it assumes
 * that the adjacency lists in both A and B are sorted in
 * ascending order. */
QValue CSR::differs(const CSR& B) const {
  QValue sum = 0;
  int i, j, k;
  for (i = 0; i < rows; ++i) {
    for (j = rowPtr[i], k = B.rowPtr[i];
        j < rowPtr[i + 1] && k < B.rowPtr[i + 1];) {
      QValue a = values[j];
      QValue b = B.values[k];
      if (colInd[j] == colInd[k]) {
        sum += (a - b) * (a - b);
        ++j, ++k;
      } else if (colInd[j] < colInd[k]){
        sum += a * a;
        ++j;
      } else {
        sum += b * b;
        ++k;
      }
    }
    for (; j < rowPtr[i + 1]; ++j) {
      sum += values[j] * values[j];
    }
    for (; k < rowPtr[i + 1]; ++k) {
      sum += B.values[k] * B.values[k];
    }
  }
  return sum;
}

vector<int> CSR::nnzStats() const {
  std::vector<int> stats(30, 0);
  for (int i = 0; i < rows; ++i) {
    long stat = rowPtr[i + 1] - rowPtr[i];
    pushToStats(rowPtr[i + 1] - rowPtr[i], stats);
  }
  return stats;
}

CSR CSR::ompRmclOneStep(const CSR &B, thread_data_t *thread_datas, const int stride) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  QValue* C;
  int nnzC;
  omp_CSR_RMCL_OneStep(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols, thread_datas, stride);
  CSR csr(C, JC, IC, this->rows, B.cols, nnzC);
  return csr;
}

CSR CSR::staticOmpRmclOneStep(const CSR &B, thread_data_t *thread_datas, const int stride) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  QValue* C;
  int nnzC;
  static_omp_CSR_RMCL_OneStep(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols, thread_datas, stride);
  CSR csr(C, JC, IC, this->rows, B.cols, nnzC);
  return csr;
}

CSR CSR::hybridOmpRmclOneStep(const CSR &B, thread_data_t *thread_datas, const int stride) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  QValue* C;
  int nnzC;
  hybrid_omp_CSR_RMCL_OneStep(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols, thread_datas, stride);
  CSR csr(C, JC, IC, this->rows, B.cols, nnzC);
  return csr;
}

CSR CSR::staticFairRmclOneStep(const CSR &B, const int stride) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  QValue* C;
  int nnzC;
  static_fair_CSR_RMCL_OneStep(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols, stride);
  CSR csr(C, JC, IC, this->rows, B.cols, nnzC);
  return csr;
}

//Input A and B matrix are one based index. Output C is also one based index.
#ifdef enable_mkl
CSR CSR::mklRmclOneStep(const CSR &B, const int stride) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  QValue* C;
  int nnzC;
  mkl_CSR_RMCL_OneStep(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols, stride);
  CSR csr(C, JC, IC, this->rows, B.cols, nnzC);
  return csr;
}
#endif


#ifdef enable_cilk
CSR CSR::cilkRmclOneStep(const CSR &B, thread_data_t *thread_datas, const int stride) const {
  assert(this->cols == B.rows);
  int* IC;
  int* JC;
  QValue* C;
  int nnzC;
  cilk_CSR_RMCL_OneStep(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      IC, JC, C, nnzC,
      this->rows, this->cols, B.cols, thread_datas, stride);
  CSR csr(C, JC, IC, this->rows, B.cols, nnzC);
  return csr;
}
#endif

#ifdef enable_gpu
CSR CSR::toGpuCSR() const {
  CSR dA;
  dA.rows = this->rows;
  dA.cols = this->cols;
  dA.nnz = this->nnz;
  cudaMalloc((void**)&dA.rowPtr, sizeof(int) * (rows + 1));
  cudaMemcpy(dA.rowPtr, rowPtr, sizeof(int) * (rows + 1), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&dA.colInd, sizeof(int) * nnz);
  cudaMemcpy(dA.colInd, colInd, sizeof(int) * nnz, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&dA.values, sizeof(QValue) * nnz);
  cudaMemcpy(dA.values, values, sizeof(QValue) * nnz, cudaMemcpyHostToDevice);
  return dA;
}
#endif

#ifdef enable_gpu
CSR CSR::toCpuCSR() const {
  CSR hA;
  hA.rows = this->rows;
  hA.cols = this->cols;
  hA.nnz = this->nnz;
  hA.rowPtr = (int*)qmalloc(sizeof(int) * (rows + 1), __FUNCTION__, __LINE__);
  HANDLE_ERROR(cudaMemcpy(hA.rowPtr, rowPtr, sizeof(int) * (rows + 1), cudaMemcpyDeviceToHost));
  hA.colInd = (int*)qmalloc(sizeof(int) * nnz, __FUNCTION__, __LINE__);
  HANDLE_ERROR(cudaMemcpy(hA.colInd, colInd, sizeof(int) * nnz, cudaMemcpyDeviceToHost));
  hA.values = (QValue*)qmalloc(sizeof(QValue) * nnz, __FUNCTION__, __LINE__);
  HANDLE_ERROR(cudaMemcpy(hA.values, values, sizeof(QValue) * nnz, cudaMemcpyDeviceToHost));
  return hA;
}
#endif

#ifdef enable_gpu
void CSR::deviceDispose() {
  cudaFree(values); values = NULL;
  cudaFree(colInd); colInd = NULL;
  cudaFree(rowPtr); rowPtr = NULL;
}
#endif

vector<int> CSR::differsStats(const CSR& B, const vector<QValue> percents) const {
  vector<int> counts(percents.size() + 4, 0);
  const int PINFI = percents.size() + 1;
  const int ZEROS = PINFI + 1;
  const int EQUALS = ZEROS + 1;
  for (int i = 0; i < rows; ++i) {
    int acount = rowPtr[i + 1] - rowPtr[i];
    int bcount = B.rowPtr[i + 1] - B.rowPtr[i];
    if (acount == 0 && bcount > 0) {
      ++counts[PINFI];
    } else if (acount == 0 && bcount == 0) {
      ++counts[ZEROS];
    } else if (acount == bcount) {
      ++counts[EQUALS];
    } else {
      QValue percent = (bcount - acount) / (QValue)acount;
      int k;
      for (k = 0; k < percents.size(); ++k) {
        if (percent < percents[k]) {
          ++counts[k];
          break;
        }
      }
      if (k == percents.size()) {
        ++counts[percents.size()];
      }
    }
  }
  int countSum = 0;
  for (int k = 0; k < counts.size(); ++k) {
    countSum += counts[k];
  }
  assert(countSum == rows);
  return counts;
}

long long CSR::spMMFlops(const CSR &B) const {
  return getSpMMFlops(this->rowPtr, this->colInd, this->values, this->nnz,
      B.rowPtr, B.colInd, B.values, B.nnz,
      this->rows, this->cols, B.cols);
}

void CSR::outputSpMMStats(const CSR &B) const {
  long long flops = this->spMMFlops(B);
  CSR C = this->omp_spmm(B, 512);
  int cNnz = C.nnz;
  C.dispose();
  printf("flops=%lld\tcNnz=%d\trows=%d\tflops/rows=%lf cnnz/rows=%lf\n", flops, cNnz, rows, (QValue)(flops) / rows, (QValue)(cNnz) / rows);
}

CSR CSR::PM(const int P[]) const {
  int* browPtr = (int*)qmalloc((rows + 1) * sizeof(int), __FUNCTION__, __LINE__);
	QValue* bvalues = (QValue*)qmalloc(nnz * sizeof(QValue), __FUNCTION__, __LINE__);
  int* bcolInd = (int*)qmalloc(nnz * sizeof(int), __FUNCTION__, __LINE__);
  browPtr[0] = 0;
  for (int i = 0; i < rows; ++i) {
    int target = P[i];
    int count = rowPtr[target + 1] - rowPtr[target];
    memcpy(bvalues + browPtr[i], values + rowPtr[target], count * sizeof(QValue));
    memcpy(bcolInd + browPtr[i], colInd + rowPtr[target], count * sizeof(int));
    browPtr[i + 1] = browPtr[i] + count;
  }
  CSR pM(bvalues, bcolInd, browPtr, rows, cols, nnz);
  return pM;
}

CSR CSR::MP(const int P[]) const {
  int* browPtr = (int*)qmalloc((rows + 1) * sizeof(int), __FUNCTION__, __LINE__);
  memcpy(browPtr, rowPtr, (rows + 1) * sizeof(int));
	QValue* bvalues = (QValue*)qmalloc(nnz * sizeof(QValue), __FUNCTION__, __LINE__);
  memcpy(bvalues, values, nnz * sizeof(QValue));
  int* bcolInd = (int*)qmalloc(nnz * sizeof(int), __FUNCTION__, __LINE__);
  for (int i = 0; i < rows; ++i) {
    for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j) {
      bcolInd[j] = P[colInd[j]];
#ifdef DEBUG
      printf("row %d %d->%d\t", i, colInd[j], P[colInd[j]]);
      printf("%d %d %lf\n", i, bcolInd[j], bvalues[j]);
#endif
    }
  }
  CSR mP(bvalues, bcolInd, browPtr, rows, cols, nnz);
  return mP;
}

CSR CSR::PMPt(const int P[]) const {
  CSR pm = PM(P);
  int *Pt = permutationTranspose(P, rows);
  CSR pMPt = pm.MP(Pt);
  pm.dispose();
  free(Pt);
  return pMPt;
}

CSR CSR::PtMP(const int P[]) const {
  CSR mP = MP(P);
  int *Pt = permutationTranspose(P, rows);
  CSR ptMP = mP.PM(Pt);
  mP.dispose();
  free(Pt);
  return ptMP;
}

int* CSR::rowDescendingOrderPermutation() {
  int* rowNnzs = (int*)malloc(rows * sizeof(int));
  int* permu = (int*)malloc(rows * sizeof(int));
  for (int i = 0; i < rows; ++i) {
    rowNnzs[i] = rowPtr[i + 1] - rowPtr[i];
    permu[i] = i;
  }
  key_value_qsort(rowNnzs, permu, rows, &(greaterThanFunction<int>));
  free(rowNnzs);
  return permu;
}

