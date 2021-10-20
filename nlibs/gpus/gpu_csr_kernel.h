#ifndef GPU_CSR_KERNEL_H
#define GPU_CSR_KERNEL_H
#include "CSR.h"

void gpuRmclIter(const int maxIter, const CSR Mgt, CSR &Mt);
CSR gpuSpMMWrapper(const CSR &dA, const CSR &dB);
void gpuOutputCSRWrapper(const CSR dA, const char *msg);
#endif
