#ifndef BCSR_H_
#define BCSR_H_
#include "CSR.h"
#include "tools/util.h"

struct BCSR {
  int r, c;
  int m, n;
  int bm, bn;
  int nnz;

	QValue* values;
	int* colInd;
	int* rowPtr;
  BCSR(const CSR &csr, const int c, const int r);

  BCSR(int m, int n, int r, int c) {
    this->m = m;
    this->n = n;
    this->r = r; this->c = c;
    bm = (m + r - 1) / r;
    bn = (n + c - 1) / c;
    values = NULL;
    colInd = NULL;
    rowPtr = NULL;
    nnz = 0;
  }

  void output(const char* msg, bool isZeroBased = true) const {
    printf("%s\n", msg);
    arrayOutput("rowPtr", stdout, rowPtr, bm + 1);
    int nnzb = rowPtr[bm];
    arrayOutput("colInd", stdout, colInd, nnzb);
    arrayOutput("values", stdout, values, nnzb * r * c);
    for (int bi = 0; bi < bm; ++bi) {
      for (int bj = rowPtr[bi]; bj < rowPtr[bi + 1]; bj++) {
        int bcol = colInd[bj];
        printf("%d\t%d\n", bi, bcol);
        QValue* bvalues = values + bj * r * c;
        for (int ei = 0; ei < r; ++ei) {
          for (int ej = 0; ej < c; ++ej) {
            QValue val = bvalues[ei * c + ej];
            printf("%.6lf\t", val);
          }
          printf("\n");
        }
        printf("\n");
      }
    }
  }

  void dispose() {
    //free(values);
    _mm_free(values);
    free(colInd);
    free(rowPtr);
  }

  bool isEqual(const CSR &B) const;

  double nonzeroDensity() const {
    return (double)nnz / (rowPtr[bm] * r * c) * 100.0;
  }
};
#endif
