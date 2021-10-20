#ifndef MCSR_H_
#define MCSR_H_
#include "CSR.h"
#include "BCSR.h"

struct MCSR : public CSR , BCSR {
  MCSR(const CSR &csr, const int c, const int r,
      const int blockRows, const int blockCols);
};
#endif
