#include "MCSR.h"

MCSR::MCSR(const CSR &csr, const int r, const int c,
    const int blockRows, const int blockCols)
: BCSR(blockRows, blockCols, r, c) {
  rows = csr.rows; cols = csr.cols;
  bool* xb = (bool*)malloc(sizeof(int) * blockCols);
  memset(xb, 0, blockCols);
  int* iJC = (int*)malloc(sizeof(int) * blockCols);
  BCSR::rowPtr = (int*)malloc((bm + 1) * sizeof(int));
  BCSR::rowPtr[0] = 0;
  CSR::rowPtr = (int*)malloc((rows + 1) * sizeof(int));
  CSR::rowPtr[0] = 0;
  int nnz = 0, nnzb = 0;
  assert(blockRows % r == 0);
  for (int it = 0; it < blockRows; it += r) {
    int topb = 0, top = 0;
    for (int i = it; i < std::min(it + r, blockRows); ++i) {
      for (int j = csr.rowPtr[i]; j < csr.rowPtr[i + 1]; ++j) {
        int col = csr.colInd[j];
        if (col < blockCols) {
          int target = col / c;
          if (!xb[target]) {
            xb[target] = true;
            iJC[topb++] = target;
          }
        } else {
          ++top;
        }
      }
      nnz += top;
      CSR::rowPtr[i + 1] = nnz;
    }
    for (int k = 0; k < topb; ++k) {
      int target = iJC[k];
      xb[target] = false;
    }
    nnzb += topb;
    BCSR::rowPtr[it / r + 1] = nnzb;
  }
  for (int i = blockRows; i < rows; ++i) {
    nnz += csr.rowPtr[i + 1] - csr.rowPtr[i];
    CSR::rowPtr[i + 1] = nnz;
  }
  arrayOutput("BCSR rowPtr", stdout, BCSR::rowPtr, blockRows + 1);
  arrayOutput("CSR rowPtr", stdout, CSR::rowPtr, rows + 1);
  assert(nnzb == BCSR::rowPtr[m]);
  printf("nnzb=%d r=%d c=%d\n", nnzb, r, c);
  printf("nnz=%d\n", nnz);
  BCSR::colInd = (int*)malloc(nnzb * sizeof(int) * r * c);
  BCSR::values = (QValue*)malloc(nnzb * sizeof(QValue) * r * c);
  memset(BCSR::values, 0, nnzb * sizeof(QValue) * r * c);
  assert(nnz == CSR::rowPtr[rows]);
  CSR::colInd = (int*)malloc(nnz * sizeof(int));
  CSR::values = (QValue*)malloc(nnz * sizeof(QValue));
  int *index = (int*)xb;
  memset(index, -1, blockCols * sizeof(int));
  int topb = 0, top = 0;
  for (int it = 0; it < blockRows; it += r) {
    for (int i = it; i < std::min(it + r, blockRows); ++i) {
      for (int j = csr.rowPtr[i]; j < csr.rowPtr[i + 1]; ++j) {
        int col = csr.colInd[j];
        if (col < blockCols) {
          int target = col / c;
          QValue* bval = NULL;
          if (index[target] == -1) {
            BCSR::colInd[topb] = target;
            bval = BCSR::values + topb * r * c;
            index[target] = topb++;
          } else {
            int pos = index[target];
            bval = BCSR::values + pos * r * c;
          }
          bval[(i % r) * c + col % c] = csr.values[j];
        } else {
          printf("top=%d col=%d\n", top, col);
          CSR::colInd[top] = col;
          CSR::values[top++] = csr.values[j];
        }
      }
    }
    for (int k = BCSR::rowPtr[it / r]; k < BCSR::rowPtr[it / r + 1]; ++k) {
      int bcol = BCSR::colInd[k];
      index[bcol] = -1;
    }
  }
  for (int i = blockRows; i < rows; ++i) {
    for (int j = csr.rowPtr[i]; j < csr.rowPtr[i + 1]; ++j) {
      CSR::values[top] = csr.values[j];
      CSR::colInd[top++] = csr.colInd[j];
    }
  }
  assert(nnz == top);
  arrayOutput("BCSR colInd", stdout, BCSR::colInd, nnzb);
  arrayOutput("CSR colInd", stdout, CSR::colInd, nnz);
  CSR::output("CSR");
}
