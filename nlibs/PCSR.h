#ifndef PCSR_H_
#define PCSR_H_
#include "CSR.h"

struct PCSR {
public:
  int rows, cols;
  int c;
  CSR *blocks;
  PCSR(const CSR &csr, const int c);
  void output(const char* msg) const {
    char msgBuffer[200];
    int stride = (cols + c - 1) / c;
    for (int b = 0; b < c; ++b) {
      sprintf(msgBuffer, "%s block %d", msg, b);
      blocks[b].coloffset_output(msgBuffer, b * stride);
    }
  }

  int stride() const {
    return (cols + c - 1) / c;
  }

  int nnz() const {
    int tnnz = 0;
    for (int b = 0; b < c; ++b) {
      tnnz += blocks[b].nnz;
    }
    return tnnz;
  }

  PCSR(const int rows, const int cols, const int c) {
    this->c = c;
    blocks = (CSR*)malloc(c * sizeof(CSR));
    memset(blocks, 0, c * sizeof(CSR));
    this->rows = rows; this->cols = cols;
  }

  void dispose() {
    if (c == 1 || c >= 2 && blocks[1].rowPtr - blocks[0].rowPtr == rows + 1) {
      free(blocks[0].rowPtr);
      free(blocks[0].colInd);
      free(blocks[0].values);
    } else {
      for (int b = 0; b < c; ++b) {
        blocks[b].dispose();
      }
    }
    free(blocks);
  }

  bool isEqual(const CSR &B) const {
    bool flag = true;
    if (rows != B.rows) {
      printf("rows = %d\tB_rows = %d\n", rows, B.rows);
      flag = false;
    }
    if (cols != B.cols) {
      printf("cols = %d\tB_cols = %d\n", cols, B.cols);
      flag = false;
    }
    int tnnz = nnz();
    if (tnnz != B.nnz) {
      printf("nnz = %d\tB_nnz = %d\n", tnnz, B.nnz);
      flag = false;
    }
    double* rowVals = (double*)malloc(cols * sizeof(double));
    memset(rowVals, 0, cols * sizeof(double));
    for (int i = 0; i < rows && flag != false; ++i) {
      int rowiNnz = 0;
      for (int b = 0; b < c; ++b) {
        rowiNnz += blocks[b].rowPtr[i + 1] - blocks[b].rowPtr[i];
      }
      int BrowiNnz = B.rowPtr[i + 1] - B.rowPtr[i];
      if (rowiNnz != BrowiNnz) {
        printf("rowNnz[%d] %d\t%d\n", i, rowiNnz, BrowiNnz);
        flag = false;
        break;
      }
      for (int b = 0; b < c; ++b) {
        for (int j = blocks[b].rowPtr[i]; j < blocks[b].rowPtr[i + 1]; ++j) {
          int col = blocks[b].colInd[j] + b * stride();
          QValue val = blocks[b].values[j];
          rowVals[col] = val;
        }
      }
      for (int j = B.rowPtr[i]; j < B.rowPtr[i + 1]; ++j) {
        int col = B.colInd[j];
        if (fabs(rowVals[col] - B.values[j]) > 1e-7) {
          printf("values[%d, %d] %lf\t%lf\n", i, col, rowVals[i], B.values[i]);
          flag = false;
          break;
        } else {
          rowVals[col] = 0.0;
        }
      }
    }
    free(rowVals);
    return flag;
  }
};

#endif
