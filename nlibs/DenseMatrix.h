#include "CSR.h"

struct DenseMatrix {
  QValue* values;
  int rows, cols;
  DenseMatrix(const CSR &csr) {
    rows = csr.rows; cols = csr.cols;
    values = (QValue*)mkl_malloc(rows * cols * sizeof(QValue), 64);
#pragma omp parallel for schedule(static, 512)
    for (int i = 0; i < rows * cols; i++) {
      values[i] = 0.0;
    }
#pragma omp parallel for schedule(static, 512)
    for (int i = 0; i < rows; i++) {
      for (int j = csr.rowPtr[i]; j < csr.rowPtr[i+1]; j++) {
        int col = csr.colInd[j];
        QValue val = csr.values[j];
        values[i * cols + col] = val;
      }
    }
  }

  DenseMatrix(const int irows, const int icols) {
    rows = irows; cols = icols;
    values = (QValue*)mkl_malloc(rows * cols * sizeof(QValue), 64);
#pragma omp parallel for schedule(static, 512)
    for (int i = 0; i < rows * cols; i++) {
      values[i] = 0.0;
    }
  }

  void output(const char* msg) {
    printf("%s\n", msg);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        printf("%lf\t", values[i * cols + j]);
      }
      printf("\n");
    }
  }

  void dispose() {
    mkl_free(values);
  }
};
