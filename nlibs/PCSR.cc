#include "PCSR.h"

PCSR::PCSR(const CSR &csr, const int c) {
    this->c = c;
    blocks = (CSR*)malloc(c * sizeof(CSR));
    memset(blocks, 0, c * sizeof(CSR));
    rows = csr.rows; cols = csr.cols;
    int stride = (cols + c - 1) / c;
    QValue *values = (QValue*)malloc(csr.nnz * sizeof(QValue));
    int *rowPtr = (int*)malloc((rows + 1) * sizeof(int) * c);
    int *colInd = (int*)malloc(csr.nnz * sizeof(int));
//#pragma omp parallel for schedule(static, 512)
    for (int i = 0; i < rows; i++) {
      for (int j = csr.rowPtr[i]; j < csr.rowPtr[i+1]; j++) {
        int col = csr.colInd[j];
        for (int b = 0; b < c; ++b) {
          if (col < stride * b + stride) {
            ++blocks[b].nnz;
            break;
          }
        }
      }
    }
    int *blockPtr = (int*)malloc((c + 1) * sizeof(int));
    blockPtr[0] = 0;
    for (int b = 0; b < c; ++b) {
      blockPtr[b + 1] = blockPtr[b] + blocks[b].nnz;
    }
    for (int b = 0; b < c; ++b) {
      blocks[b].init(values + blockPtr[b], colInd + blockPtr[b], rowPtr + b * (rows + 1),
          rows, stride, blocks[b].nnz);
    }
    assert(blockPtr[c] == csr.rowPtr[rows]);
    free(blockPtr);
    for (int b = 0; b < c; ++b) {
      blocks[b].rowPtr[0] = 0;
    }
//#pragma omp parallel for schedule(static, 512)
    for (int i = 0; i < rows; i++) {
      for (int b = 0; b < c; ++b) {
        blocks[b].rowPtr[i + 1] = blocks[b].rowPtr[i];
      }
      for (int j = csr.rowPtr[i]; j < csr.rowPtr[i + 1]; j++) {
        int col = csr.colInd[j];
        QValue val = csr.values[j];
        for (int b = 0; b < c; ++b) {
          if (col < stride * b + stride) {
            blocks[b].values[blocks[b].rowPtr[i + 1]] = val;
            blocks[b].colInd[blocks[b].rowPtr[i + 1]] = col - stride * b;
            ++blocks[b].rowPtr[i + 1];
            break;
          }
        }
      }
    }
}
