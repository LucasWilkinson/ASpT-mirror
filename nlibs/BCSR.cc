#include "BCSR.h"

BCSR::BCSR(const CSR &csr, const int r, const int c) {
    m = csr.rows; bm = (m + r - 1) / r;
    n = csr.cols; bn = (n + c - 1) / c;
    this->r = r; this->c = c;
    assert(csr.nnz == csr.rowPtr[m]);
    nnz = csr.nnz;
    bool* xb = (bool*)malloc(sizeof(int) * bn);
    memset(xb, 0, bn);
    int* iJC = (int*)malloc(sizeof(int) * bn);
    int nnzb = 0;
    rowPtr = (int*)malloc((bm + 1) * sizeof(int));
    rowPtr[0] = 0;
    for (int it = 0; it < m; it += r) {
      int top = 0;
      for (int i = it; i < std::min(it + r, m); ++i) {
        for (int j = csr.rowPtr[i]; j < csr.rowPtr[i + 1]; ++j) {
          int col = csr.colInd[j];
          int target = col / c;
          if (!xb[target]) {
            xb[target] = true;
            iJC[top++] = target;
          }
        }
      }
      for (int k = 0; k < top; ++k) {
        int target = iJC[k];
        xb[target] = false;
      }
      nnzb += top;
      rowPtr[it / r + 1] = nnzb;
    }
    values = (QValue*)_mm_malloc(nnzb * sizeof(QValue) * r * c, 4096);
    memset(values, 0, nnzb * sizeof(QValue) * r * c);
    colInd = (int*)malloc(nnzb * sizeof(int));
    memset(colInd, -1, nnzb * sizeof(int));
    int *index = (int*)xb;
    memset(index, -1, bn * sizeof(int));
    int top = 0;
    for (int it = 0; it < m; it += r) {
      for (int i = it; i < std::min(it + r, m); ++i) {
        for (int j = csr.rowPtr[i]; j < csr.rowPtr[i + 1]; ++j) {
          int col = csr.colInd[j];
          int target = col / c;
          QValue* bval = NULL;
          if (index[target] == -1) {
            colInd[top] = target;
            bval = values + top * r * c;
            index[target] = top++;
          } else {
            int pos = index[target];
            bval = values + pos * r * c;
          }
          bval[(i % r) * c + col % c] = csr.values[j];
        }
      }
      for (int k = rowPtr[it / r]; k < rowPtr[it / r + 1]; ++k) {
        int target = colInd[k];
        index[target] = -1;
      }
    }
    free(xb);
    free(iJC);
}

bool BCSR::isEqual(const CSR &B) const {
  int* iJC = (int*)malloc(n * sizeof(int));
  memset(iJC, 0, n * sizeof(int));
  bool* xb = (bool*)malloc(n);
  memset(xb, 0, n);
  double* x = (double*)malloc(n * sizeof(double));
  memset(x, 0, n * sizeof(double));
  bool flag = true;
  for (int bi = 0; bi < bm && flag; ++bi) {
    for (int ei = 0; ei < r && bi * r + ei < m && flag; ++ei) {
      int top = 0;
      int row = bi * r + ei;
      for (int bj = rowPtr[bi]; bj < rowPtr[bi + 1]; ++bj) {
        int bcol = colInd[bj];
        for (int ej = 0; ej < c && bcol * c + ej < n; ++ej) {
          int col = bcol * c + ej;
          QValue val = values[bj * r * c + ei * c + ej];
          if (!xb[col] && fabs(val) > 1e-9) {
            xb[col] = true;
            x[col] = val;
            iJC[top++] = col;
          }
        }
      }
      for (int Bj = B.rowPtr[row]; Bj < B.rowPtr[row + 1]; ++Bj) {
        int col = B.colInd[Bj];
        QValue val = B.values[Bj];
        if (fabs(x[col] - val) > 1e-9) {
          printf("row=%d col=%d val %lf VS B %lf\n", row, col, x[col], val);
          flag = false;
          break;
        }
      }
      if (B.rowPtr[row + 1] - B.rowPtr[row] != top) {
        printf("row %d: %d VS B %d\n", row, top, B.rowPtr[row + 1] - B.rowPtr[row]);
        flag = false;
        break;
      }
      for (int k = 0; k < top; ++k) {
        int col = iJC[k];
        xb[col] = false;
        x[col] = 0.0;
      }
    }
  }
  free(xb);
  free(x);
  free(iJC);
  return flag;
}

