#include "cusparse_spmm.h"
#include "gpus/cuda_handle_error.h"

bool isCusparseInit = false;
cusparseStatus_t status;
cusparseHandle_t handle  = 0;
cusparseMatDescr_t descrA = 0;
cusparseMatDescr_t descrB = 0;
cusparseMatDescr_t descrC = 0;
int cusparse_init(void) {
    /* initialize cusparse library */
    status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("CUSPARSE Library initialization failed");
        return 1;
    }

    /* create and setup matrix descriptor */
    status = cusparseCreateMatDescr(&descrA);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Matrix descriptor initialization failed");
        return 1;
    }
    cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);

    status = cusparseCreateMatDescr(&descrB);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Matrix descriptor initialization failed");
        return 1;
    }
    cusparseSetMatType(descrB,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrB,CUSPARSE_INDEX_BASE_ZERO);

    status = cusparseCreateMatDescr(&descrC);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Matrix descriptor initialization failed");
        return 1;
    }
    cusparseSetMatType(descrC,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrC,CUSPARSE_INDEX_BASE_ZERO);

    return 0;
}

void gpuCsrSpMM(const int dIA[], const int dJA[], const QValue dA[], const int nnzA,
        const int dIB[], const int dJB[], const QValue dB[], const int nnzB,
        int* &dIC, int* &dJC, QValue* &dC, int& nnzC,
        const int m, const int k, const int n) {
}

void cusparseXcsrgemmNnzWrapper(const int dIA[], const int dJA[], const int nnzA,
        const int dIB[], const int dJB[], const int nnzB,
        const int m, const int k, const int n,
        int* IC, int& nnzC) {
  cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
  status = cusparseXcsrgemmNnz(handle, transA, transB, m, n, k,
      descrA, nnzA, dIA, dJA,
      descrB, nnzB, dIB, dJB,
      descrC, IC, &nnzC);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    CLEANUP("CSR Matrix-Matrix multiplication failed");
  }
}

void cusparseDcsrgemmWapper(const int* const dIA, const int dJA[], const QValue dA[], const int nnzA,
        const int dIB[], const int dJB[], const QValue dB[], const int nnzB,
        const int* dIC, int* dJC, QValue* dC, const int nnzC,
        const int m, const int k, const int n) {
  cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
#ifdef FDOUBLE
  status = cusparseDcsrgemm(
      handle, transA, transB, m, n, k,
      descrA, nnzA,
      dA, dIA, dJA,
      descrB, nnzB,
      dB, dIB, dJB,
      descrC,
      dC, dIC, dJC);
#else
  status = cusparseScsrgemm(
      handle, transA, transB, m, n, k,
      descrA, nnzA,
      dA, dIA, dJA,
      descrB, nnzB,
      dB, dIB, dJB,
      descrC,
      dC, dIC, dJC);
#endif
  if (status != CUSPARSE_STATUS_SUCCESS) {
    CLEANUP("CSR Matrix-Matrix multiplication failed");
  }
}

CSR cusparseSpMMWrapper(const CSR &dA, const CSR &dB) {
  //cusparse_init();
  int m = dA.rows;
  int k = dA.cols;
  int n = dB.cols;
  CSR dC;
  int baseC, nnzC;
  dC.rows = m; dC.cols = n;
  timer t;
  cudaMalloc((void**)&dC.rowPtr, sizeof(int) * (m + 1));
  timer t2;
  cusparseXcsrgemmNnzWrapper(dA.rowPtr, dA.colInd, dA.nnz,
      dB.rowPtr, dB.colInd, dB.nnz,
      m, k, n,
      dC.rowPtr, nnzC);
  double nnzTime = t2.milliseconds_elapsed();
  HANDLE_ERROR(cudaMemcpy(&nnzC , dC.rowPtr + m, sizeof(int), cudaMemcpyDeviceToHost));
  cudaMemcpy(&baseC, dC.rowPtr, sizeof(int), cudaMemcpyDeviceToHost);
  nnzC -= baseC;
  cudaMalloc((void**)&dC.colInd, sizeof(int) * nnzC);
  cudaMalloc((void**)&dC.values, sizeof(QValue) * nnzC);
  dC.nnz = nnzC;
  cusparseDcsrgemmWapper(dA.rowPtr, dA.colInd, dA.values, dA.nnz,
      dB.rowPtr, dB.colInd, dB.values, dB.nnz,
      dC.rowPtr, dC.colInd, dC.values, dC.nnz,
      m, k, n);
  cudaDeviceSynchronize();
  printf("cusparse time passed %lf nnzTime=%lf\n", t.milliseconds_elapsed(), nnzTime);
  //cusparse_finalize("clear up cusparse");
  return dC;
}

void cusparse_finalize(const char *msg) {
  CLEANUP(msg);
}
