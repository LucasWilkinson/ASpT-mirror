/*
 * CSC.h
 *
 *  Created on: Aug 8, 2017
 *      Author: vineeth
 */

#ifndef CSC_H_
#define CSC_H_
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "math.h"
#include "cpu_csc_kernel.h"
#include "tools/key_value_qsort.h"
#ifdef enable_gpu
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif
#include <vector>
using namespace std;

struct CSC {
public:
/* A real or complex array that contains the non-zero elements of a sparse matrix.
 * The non-zero elements are mapped into the values array using the row-major upper
 * triangular storage mapping described above.*/
	QValue* values;

/* Element i of the integer array columns is the number of the rows that
 * contain the i-th element in the values array.*/
	int* rowInd;

/* Element j of the integer array colIndex gives the index of the element
 * in the values array that is
 * first non-zero element in a col j.*/
	int* colPtr;
	int rows, cols, nnz;

/* if allocate by malloc, isStatus=1
 * else if allocate by cudaMalloc isStatus=-1
 * else isStatus=0 */
	CSC() {
    this->values = NULL;
    this->rowInd = NULL;
    this->colPtr = NULL;
    this->rows = 0;
    this->cols = 0;
    this->nnz = 0;
	}

  CSC deepCopy();

  void initWithDenseMatrix(const QValue* dvalues, const int rows, const int cols) {
    this->rows = rows; this->cols = cols;
    colPtr = (int*)malloc((cols + 1) * sizeof(int));
    colPtr[0] = 0;
    nnz = 0;
    for (int i = 0; i < cols; ++i) {
      for (int j = 0; j < rows; ++j) {
        QValue val = dvalues[i * rows + j];
        if (val < 1e-8) {
          continue;
        }
        ++nnz;
      }
      colPtr[i + 1] = nnz;
    }
    rowInd = (int*)malloc(nnz * sizeof(int));
    values = (QValue*)malloc(nnz * sizeof(QValue));
    int top = 0;
    for (int i = 0; i < cols; ++i) {
      for (int j = 0; j < rows; ++j) {
        QValue val = dvalues[i * rows + j];
        if (val < 1e-8) {
          continue;
        }
        rowInd[top] = j;
        values[top++] = val;
      }
    }
  }

	void init(QValue* values, int* rowInd, int* colPtr, int rows, int cols, int nnz) {
    this->values = values;
    this->rowInd = rowInd;
    this->colPtr = colPtr;
    this->rows = rows;
    this->cols = cols;
    this->nnz = nnz;
  }

	CSC(QValue* values, int* rowInd, int* colPtr, int rows, int cols, int nnz) {
    init(values, rowInd, colPtr, rows, cols, nnz);
	}

  void averAndNormRowQValue();
  long spmmFlops(const CSC& B) const;
  std::vector<int> multiFlopsStats(const CSC& B) const;
  vector<int> nnzStats() const;
  CSC spmm(const CSC& B) const;
  CSC flops_spmm(const CSC& B, const int stride = 512) const;
  //CSC group_spmm(const CSC& B, const int stride = 512) const;
  //CSC omp_spmm(const CSC& B, const int stride = 512) const;
  //CSC omp_spmm(thread_data_t* thread_datas, const CSC& B, const int stride = 512) const;
  CSC somp_spmm(const CSC& B, const int stride = 512) const;
  //CSC somp_spmm_memory_Managed(const CSC& B, const int stride = 512) const;
  CSC somp_spmm(thread_data_t* thread_datas, const CSC& B, const int stride = 512) const;
  //CSC noindex_somp_spmm(const CSC& B, const int stride = 512) const;
  void output(const char* msg, bool isZeroBased = true) const {
    printf("%s\n", msg);
    if (isZeroBased) {
      for (int i = 0; i < cols; i++) {
        for (int j = colPtr[i]; j < colPtr[i+1]; j++) {
          int row=rowInd[j];
          QValue val=values[j];
          printf("%d\t%d\t%.6lf\n", i, row, val);
        }
      }
    } else {
      for (int i = 0; i < cols; i++) {
        for (int j = colPtr[i] - 1; j < colPtr[i+1] - 1; j++) {
          int row=rowInd[j];
          QValue val=values[j];
          printf("%d\t%d\t%.6lf\n", i + 1, row+1, val);
        }
      }
    }
  }

	void outputToFile(const char* msg, const char fname[], bool isZeroBased = true) const {
    printf("%s\n", msg);
		FILE *output_file;
	  if ((output_file = fopen(fname, "ab")) == NULL) {
	    printf("Failed to open file %s\n", fname);
	    exit(-1);
	  }
    if (isZeroBased) {
      for (int i = 0; i < cols; i++) {
        for (int j = colPtr[i]; j < colPtr[i+1]; j++) {
          int row=rowInd[j];
          QValue val=values[j];
          fprintf(output_file, "%d\t%d\t%.6lf\n", i + 1, row, val);
        }
      }
    } else {
      for (int i = 0; i < cols; i++) {
        for (int j = colPtr[i] - 1; j < colPtr[i+1] - 1; j++) {
          int row=rowInd[j];
          QValue val=values[j];
					fprintf(output_file, "%d\t%d\t%.6lf\n", i + 1, row+1, val);
        }
      }
    }
  }

  void rowoffset_output(const char* msg, int rowOffset) const {
    printf("%s\n", msg);
    for (int i = 0; i < cols; i++) {
      for (int j = colPtr[i]; j < colPtr[i+1]; j++) {
        int row = rowInd[j] + rowOffset;
        QValue val = values[j];
        printf("%d\t%d\t%.6lf\n", i, row, val);
      }
    }
  }

  void output_int(const char* msg) const {
    printf("%s\n", msg);
    for (int i = 0; i < cols; i++) {
      for (int j = colPtr[i]; j < colPtr[i+1]; j++) {
        int row=rowInd[j];
        int val=((int*)values)[j];
        printf("%d\t%d\t%d\n", i, row, val);
      }
    }
  }

  void toAbs() {
    for (int i = 0; i < cols; i++) {
      for (int j = colPtr[i]; j < colPtr[i + 1]; j++) {
        values[j] = fabs(values[j]);
      }
    }
  }

  void output_structure(const char* msg) const {
    printf("%s\n", msg);
    for (int i = 0; i < cols; i++) {
      for (int j = colPtr[i]; j < colPtr[i+1]; j++) {
        int row=rowInd[j];
        printf("%d\t%d\n", i, row);
      }
    }
  }

  void toOneBasedCSC() {
    for (int i = 0; i < cols; i++) {
      for (int j = colPtr[i]; j < colPtr[i+1]; j++) {
        ++rowInd[j];
      }
    }
    for (int i = 0; i < cols + 1; i++) {
      ++colPtr[i];
    }
  }

  void toZeroBasedCSC() {
    for (int i = 0; i < cols + 1; i++) {
      --colPtr[i];
    }
    for (int i = 0; i < cols; i++) {
      for (int j = colPtr[i]; j < colPtr[i+1]; j++) {
        --rowInd[j];
      }
    }
  }

  void makeOrdered();
  void matrixRowReorder(const int* ranks) const;

  bool isEqual(const CSC &B) const {
    bool flag = true;
    if (rows != B.rows) {
      printf("rows = %d\tB_rows = %d\n", rows, B.rows);
      flag = false;
    }
    if (cols != B.cols) {
      printf("cols = %d\tB_cols = %d\n", cols, B.cols);
      flag = false;
    }
    if (nnz != B.nnz) {
      printf("nnz = %d\tB_nnz = %d\n", nnz, B.nnz);
      flag = false;
    }

    for (int i = 0; i < (cols + 1); ++i) {
      if (colPtr[i] != B.colPtr[i]) {
        printf("colPtr[%d] %d\t%d\n", i, colPtr[i], B.colPtr[i]);
        flag = false;
        break;
      }
    }

    QValue* colVals = (QValue*)malloc(rows * sizeof(QValue));
    memset(colVals, 0, rows * sizeof(QValue));
    for (int i = 0; i < cols && flag != false; ++i) {
      for (int j = colPtr[i]; j < colPtr[i + 1]; ++j) {
        int row = rowInd[j];
        colVals[row] = values[j];
      }
      for (int j = B.colPtr[i]; j < B.colPtr[i + 1]; ++j) {
        int row = B.rowInd[j];
        if (fabs(colVals[row] - B.values[j]) > 1e-7) {
          printf("values[%d][%d] %lf\t%lf\n", i, row, colVals[row], B.values[j]);
          flag = false;
          break;
        } else {
          colVals[row] = 0.0;
        }
      }
    }
    free(colVals);
    return flag;
  }

  //RawEqual: The elements in A or B is not promising nonzero so that the number of nonzero
  //is not guaranteed equal.
  bool isRawEqual(const CSC &B) const {
    bool flag = true;
    if (rows != B.rows) {
      printf("rows = %d\tB_rows = %d\n", rows, B.rows);
      flag = false;
    }
    if (cols != B.cols) {
      printf("cols = %d\tB_cols = %d\n", cols, B.cols);
      flag = false;
    }
    
    QValue* colVals = (QValue*)malloc(rows * sizeof(QValue));
    memset(colVals, 0, rows * sizeof(QValue));
    for (int i = 0; i < cols && flag != false; ++i) {
      for (int j = colPtr[i]; j < colPtr[i + 1]; ++j) {
        int row = rowInd[j];
        if (fabs(values[j]) > 1e-8) {
          colVals[row] = values[j];
        }
      }
      for (int j = B.colPtr[i]; j < B.colPtr[i + 1]; ++j) {
        int row = B.rowInd[j];
        if (fabs(colVals[row] - B.values[j]) > 1e-8) {
          printf("values[%d, %d] %lf\t%lf\n", i, row, colVals[row], B.values[j]);
          flag = false;
          break;
        } else {
          colVals[row] = 0.0;
        }
      }
      free(colVals);
      return flag;
    }
    return flag;
  }


  bool isRelativeEqual(const CSC &B, float maxRelativeError) const {
    bool flag = true;
    if (rows != B.rows) {
      printf("rows = %d\tB_rows = %d\n", rows, B.rows);
      flag = false;
    }
    if (cols != B.cols) {
      printf("cols = %d\tB_cols = %d\n", cols, B.cols);
      flag = false;
    }
    
    QValue* colVals = (QValue*)malloc(rows * sizeof(QValue));
    memset(colVals, 0, rows * sizeof(QValue));


    for (int i = 0; i < cols && flag != false; ++i) {
      for (int j = colPtr[i]; j < colPtr[i + 1]; ++j) {
        int row = rowInd[j];
        if (fabs(values[j]) > 1e-8) {
          colVals[row] = values[j];
        }
      }
      for (int j = B.colPtr[i]; j < B.colPtr[i + 1]; ++j) {
        int row = B.rowInd[j];
        
        float relativeError = fabs((colVals[row] - B.values[j]) / B.values[j]);
        if (relativeError > maxRelativeError && fabs(B.values[j]) > 1e-8) {
          printf("values[%d, %d] %e should be %e\n", i, row, colVals[row], B.values[j]);
          flag = false;
          break;
        } else {
          colVals[row] = 0.0;
        }
      }
      free(colVals);
      return flag;
    }

    return flag;
  }

  void dispose() {
      free(values); values = NULL;
      free(rowInd); rowInd = NULL;
      free(colPtr); colPtr = NULL;
  }

  void deviceDispose();

  /*Default columnInflation gamma is 2*/
  void columnInflationR2(int colId) const {
    for (int i = colPtr[colId]; i < colPtr[colId + 1]; ++i) {
      values[i] = values[i] * values[i];
    }
  }

  inline QValue colMax(int colId) const {
    QValue rmax = 0.0;
    for (int i = colPtr[colId]; i < colPtr[colId + 1]; ++i) {
      if (rmax < values[i]) {
        rmax = values[i];
      }
    }
    return rmax;
  }

  inline QValue colSum(int colId) const {
    QValue sum = 0.0;
    for (int i = colPtr[colId]; i < colPtr[colId + 1]; ++i) {
        sum += values[i];
    }
    return sum;
  }

  inline int colCount(int colId) const {
    return colPtr[colId + 1] - colPtr[colId];
  }

//   CSC ompRmclOneStep(const CSC &B, thread_data_t *thread_datas, const int stride) const;
//   CSC mklRmclOneStep(const CSC &B, const int stride) const;
//   CSC staticOmpRmclOneStep(const CSC &B, thread_data_t *thread_datas, const int stride) const;
//   CSC hybridOmpRmclOneStep(const CSC &B, thread_data_t *thread_datas, const int stride) const;
//   CSC staticFairRmclOneStep(const CSC &B, const int stride) const;
// #ifdef enable_cilk
//   CSC cilkRmclOneStep(const CSC &B, thread_data_t *thread_datas, const int stride) const;
// #endif
   QValue differs(const CSC& B) const;
//   vector<int> differsStats(const CSC& B, vector<QValue> percents) const;
//   CSC toGpuCSC() const;
//   CSC toCpuCSC() const;
   long long spMMFlops(const CSC& B) const;
//   void outputSpMMStats(const CSC& B) const;
//   CSC PM(const int P[]) const;
//   CSC MP(const int P[]) const;
//   CSC PMPt(const int P[]) const;
//   CSC PtMP(const int P[]) const;
//   int* rowDescendingOrderPermutation();

};
#endif /* CSC_CUH_ */
