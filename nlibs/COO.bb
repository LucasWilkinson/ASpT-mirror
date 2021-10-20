/*
 * COO.cpp
 *
 *  Created on: Oct 3, 2013
 *      Author: niuq
 */

#include "COO.h"
#include "tools/qmalloc.h"
//#include <tuple>
#include <vector>
#include <algorithm>

COO::COO() {
	cooRowIndex=cooColIndex=NULL;
	cooVal=NULL;
	nnz=0; rows=cols=0;
}

COO::COO(const char fname[]) {
  this->readMatrixMarketFile(fname);
}

COO::COO(const QValue* const cooVal, const int* const cooColIndex,
      const int* const cooRowIndex, const int rows, const int cols, const int nnz) {
  this->rows = rows;
  this->cols = cols;
  this->nnz = nnz;
  this->cooColIndex = (int*)qmalloc(nnz * sizeof(int), __FUNCTION__, __LINE__);
  this->cooRowIndex = (int*)qmalloc(nnz * sizeof(int), __FUNCTION__, __LINE__);
  this->cooVal = (QValue*)qmalloc(nnz * sizeof(QValue), __FUNCTION__, __LINE__);
  memcpy(this->cooColIndex, cooColIndex, nnz * sizeof(int));
  memcpy(this->cooRowIndex, cooRowIndex, nnz * sizeof(int));
  memcpy(this->cooVal, cooVal, nnz * sizeof(QValue));
}

void COO::dispose() {
	free(cooRowIndex); cooRowIndex = NULL;
	free(cooColIndex); cooColIndex = NULL;
	free(cooVal); cooVal = NULL;
}

void COO::readMatrixMarketFile(const char fname[]) {
	mm_read_unsymmetric_sparse(fname, &rows, &cols, &nnz,
            &cooVal, &cooRowIndex, &cooColIndex);
}

int COO::readSNAPFile(const char fname[], bool isTrans) {
  FILE *fpin;
  if ((fpin = fopen(fname, "r")) == NULL) {
    printf("Failed to open file %s\n", fname);
    exit(-1);
  }
  char line[MM_MAX_LINE_LENGTH];
  char banner[MM_MAX_TOKEN_LENGTH];
  char mtx[MM_MAX_TOKEN_LENGTH]; mtx[0] = '\0';
  char crd[MM_MAX_TOKEN_LENGTH];
  char data_type[MM_MAX_TOKEN_LENGTH];
  char storage_scheme[MM_MAX_TOKEN_LENGTH];
  strcpy(storage_scheme, "unsymmetric");
  char *p;
  fgets(line, MM_MAX_LINE_LENGTH, fpin);
  bool isMtx = false;
  if (line[0] == '%' && !feof(fpin)) {
    if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type, storage_scheme) == 5) {
      for (p = mtx; *p != '\0'; *p = tolower(*p), p++);  /* convert to lower case */
      for (p = crd; *p != '\0'; *p = tolower(*p), p++);
      for (p = data_type; *p != '\0'; *p = tolower(*p), p++);
      for (p = storage_scheme; *p != '\0'; *p = tolower(*p), p++);
      isMtx = true;
    }
  }
  while ((line[0] == '#' || line[0] == '%') && !feof(fpin)) {
    fgets(line, MM_MAX_LINE_LENGTH, fpin);
  }
  if (feof(fpin)) {
    nnz = 0;
    return 0;
  }
  int f2, f3;
  int ret = sscanf(line, "%d %d %d", &(this->rows), &f2, &f3);
  if (ret == 2) {
    this->cols = this->rows;
    this->nnz = f2;
  } else {
    assert (ret == 3);
    this->cols = f2;
    this->nnz = f3;
  }
  printf("rows=%d cols=%d nnz=%d\n", rows, cols, nnz);
  //printf("%s\n", banner);
  if (strcmp(storage_scheme, "symmetric") == 0) {
    cooRowIndex = (int*)qmalloc(nnz * 2 * sizeof(int), __FUNCTION__, __LINE__);
    cooColIndex = (int*)qmalloc(nnz * 2 * sizeof(int), __FUNCTION__, __LINE__);
    cooVal = (QValue*)qmalloc(nnz * 2 * sizeof(QValue), __FUNCTION__, __LINE__);
    int top = 0;
    int from, to;
    QValue val;
    printf("symm %d %d %d\n", this->rows, this->cols, this->nnz);
    for (int i = 0; i < nnz; ++i) {
#ifdef FDOUBLE
      int ret = fscanf(fpin, "%d%d%lf", &from, &to, &val);
#else
      int ret = fscanf(fpin, "%d%d%f", &from, &to, &val);
#endif
      assert (isMtx);
      --from; --to;
      //isTrans is not useful if symmetric
      assert (ret == 2 || ret == 3);
      if (ret == 2) {
        val = 1.0;
      }
      cooRowIndex[top] = from;
      cooColIndex[top] = to;
      cooVal[top++] = val;
      if (from != to) {
        cooRowIndex[top] = to;
        cooColIndex[top] = from;
        cooVal[top++] = val;
      }
    }
    this->nnz = top;
  } else {
    cooRowIndex = (int*)qmalloc(nnz * sizeof(int), __FUNCTION__, __LINE__);
    cooColIndex = (int*)qmalloc(nnz * sizeof(int), __FUNCTION__, __LINE__);
    cooVal = (QValue*)qmalloc(nnz * sizeof(QValue), __FUNCTION__, __LINE__);
    int from, to;
    QValue val;
////    printf("nonsymm %d %d %d\n", this->rows, this->cols, this->nnz);
    for (int i = 0; i < nnz; ++i) {
      fgets(line, MM_MAX_LINE_LENGTH, fpin);
#ifdef FDOUBLE
      int ret = sscanf(line, "%d%d%lf", &from, &to, &val);
#else
      int ret = sscanf(line, "%d%d%f", &from, &to, &val);
#endif
      if (isMtx) {
        assert (isMtx);
        --from; --to;
      }
      //Reverse row and col so that it is transposed.
      if (isTrans) {
        cooRowIndex[i] = to;
        cooColIndex[i] = from;
      } else {
        cooRowIndex[i] = from;
        cooColIndex[i] = to;
      }
      if (ret == 2) {
        cooVal[i] = 1.0;
      } else {
        assert (ret == 3);
        cooVal[i] = val;
      }
    }
  }
  return 0;
}

void COO::addSelfLoopIfNeeded() {
  assert(rows == cols);
  bool* u = (bool*)calloc(rows, sizeof(bool));
  int count = 0;
  for (int i = 0; i < nnz; ++i) {
    int row = cooRowIndex[i];
    int col = cooColIndex[i];
    if (row == col) {
      u[row] = true;
      count++;
    }
  }
  int needed = rows - count;
  int oldNnz = nnz;
  nnz += needed;
  cooRowIndex = (int*)realloc(cooRowIndex, nnz * sizeof(int));
  cooColIndex = (int*)realloc(cooColIndex, nnz * sizeof(int));
  cooVal = (QValue*)realloc(cooVal, nnz * sizeof(QValue));
  int top = oldNnz;
  for (int i = 0; i < rows; ++i) {
    if (u[i]) {
      continue;
    }
    cooRowIndex[top] = i;
    cooColIndex[top] = i;
    cooVal[top++] = 1.0;
  }
  free(u);
}

void COO::output(const char* msg) {
  printf("%s\n", msg);
  for(int i=0;i<nnz;i++)
  {
#ifdef FDOUBLE
    printf("%d %d %lf\n", cooRowIndex[i], cooColIndex[i], cooVal[i]);
#else
    printf("%d %d %f\n", cooRowIndex[i], cooColIndex[i], cooVal[i]);
#endif
  }
  printf("host output end\n");
}

struct COOTuple {
  int rowIndex;
  int colIndex;
  QValue val;
};

bool operator < (const COOTuple &A, const COOTuple &B) {
  return A.rowIndex < B.rowIndex || (A.rowIndex == B.rowIndex && A.colIndex < B.colIndex);
}


COOTuple makeCOOTuple(int rowIndex, int colIndex, QValue val) {
  COOTuple cooTuple;
  cooTuple.rowIndex = rowIndex;
  cooTuple.colIndex = colIndex;
  cooTuple.val = val;
  return cooTuple;
}

void COO::makeOrdered() const {
  typedef COOTuple iid;
  std::vector<iid> v;
  v.resize(nnz);
  for (int i = 0; i < nnz; ++i) {
    v[i] = makeCOOTuple(cooRowIndex[i], cooColIndex[i], cooVal[i]);
  }
  std::sort(v.begin(), v.end());
  for (int i = 0; i < nnz; ++i) {
    cooRowIndex[i] = v[i].rowIndex;
    cooColIndex[i] = v[i].colIndex;
    cooVal[i] = v[i].val;
  }
}

void COO::makeOrderedByCol() const {
  typedef COOTuple iid;
  std::vector<iid> v;
  v.resize(nnz);
  for (int i = 0; i < nnz; ++i) {
    v[i] = makeCOOTuple(cooRowIndex[i], cooColIndex[i], cooVal[i]);
  }
  std::sort(v.begin(), v.end(),[](const iid &it1, const iid &it2){return (it1.colIndex < it2.colIndex);});
  for (int i = 0; i < nnz; ++i) {
    cooRowIndex[i] = v[i].rowIndex;
    cooColIndex[i] = v[i].colIndex;
    cooVal[i] = v[i].val;
  }
}

int COO::orderedAndDuplicatesRemoving() {
  typedef COOTuple iid;
  std::vector<iid> v;
  v.resize(nnz);
  for (int i = 0; i < nnz; ++i) {
    v[i] = makeCOOTuple(cooRowIndex[i], cooColIndex[i], cooVal[i]);
  }
  std::sort(v.begin(), v.end());
  int i = 1, j = 0;
  while (i < nnz) {
    if (v[i].rowIndex == v[j].rowIndex && v[i].colIndex == v[j].colIndex) {
      v[j].val += v[i].val;
    } else {
      ++j;
      v[j] = v[i];
    }
    ++i;
  }
  nnz = j + 1;
  for (int i = 0; i < nnz; ++i) {
    cooRowIndex[i] = v[i].rowIndex;
    cooColIndex[i] = v[i].colIndex;
    cooVal[i] = v[i].val;
  }
  cooRowIndex = (int*)realloc(cooRowIndex, nnz * sizeof(int));
  cooColIndex = (int*)realloc(cooColIndex, nnz * sizeof(int));
  cooVal = (QValue*)realloc(cooVal, nnz * sizeof(QValue));
  return j + 1;
}

int COO::orderedByColAndDuplicatesRemoving() {
  typedef COOTuple iid;
  std::vector<iid> v;
  v.resize(nnz);
  for (int i = 0; i < nnz; ++i) {
    v[i] = makeCOOTuple(cooRowIndex[i], cooColIndex[i], cooVal[i]);
  }
  std::sort(v.begin(), v.end(),[](const iid &it1, const iid &it2)
    {return ((it1.colIndex==it2.colIndex)? (it1.rowIndex < it2.rowIndex):(it1.colIndex < it2.colIndex) );});
  
  int i = 1, j = 0;
  while (i < nnz) {
    if (v[i].rowIndex == v[j].rowIndex && v[i].colIndex == v[j].colIndex) {
      v[j].val += v[i].val;
    } else {
      ++j;
      v[j] = v[i];
    }
    ++i;
  }
  nnz = j + 1;
  
  for (int i = 0; i < nnz; ++i) {
    cooRowIndex[i] = v[i].rowIndex;
    cooColIndex[i] = v[i].colIndex;
    cooVal[i] = v[i].val;
  }
  cooRowIndex = (int*)realloc(cooRowIndex, nnz * sizeof(int));
  cooColIndex = (int*)realloc(cooColIndex, nnz * sizeof(int));
  cooVal = (QValue*)realloc(cooVal, nnz * sizeof(QValue));
  return j + 1;
}

//COO format must be in order
CSR COO::toCSR() const {
  int row = 0;
	int* ocsrRowPtr = (int*)qmalloc(sizeof(int) * (rows + 1), __FUNCTION__, __LINE__);
	memset(ocsrRowPtr, -1, sizeof(int) * (rows + 1));
	for (int t = 0; t < nnz; ++t) {
    while (row < cooRowIndex[t] && row < rows && ocsrRowPtr[row] == -1)
      ocsrRowPtr[row++] = t;
    if (row == cooRowIndex[t] && ocsrRowPtr[row] == -1)
      ocsrRowPtr[row++] = t;
	}
  for (; row < rows; ++row) {
    if (ocsrRowPtr[row] == -1) {
		  ocsrRowPtr[row] = nnz;
    }
  }
	ocsrRowPtr[rows] = nnz;
	int onnz = nnz;
	int* ocsrColInd = (int*)qmalloc(sizeof(int) * onnz, __FUNCTION__, __LINE__);
	QValue* ocsrVals = (QValue*)qmalloc(sizeof(QValue) * onnz, __FUNCTION__, __LINE__);
	memcpy(ocsrColInd, cooColIndex, sizeof(int) * onnz);
	memcpy(ocsrVals, cooVal, sizeof(QValue) * onnz);
	CSR csr(ocsrVals, ocsrColInd, ocsrRowPtr, rows, cols, onnz);
	return csr;
}

//COO format must be in order
CSC COO::toCSC() const {
  int col = 0;
  int* ocscColPtr = (int*)qmalloc(sizeof(int) * (cols + 1), __FUNCTION__, __LINE__);
  memset(ocscColPtr, -1, sizeof(int) * (cols + 1));
  for (int t = 0; t < nnz; ++t) {
    while (col < cooColIndex[t] && col < cols && ocscColPtr[col] == -1)
      ocscColPtr[col++] = t;
    if (col == cooColIndex[t] && ocscColPtr[col] == -1)
      ocscColPtr[col++] = t;
  }
  for (; col < cols; ++col) {
    if (ocscColPtr[col] == -1) {
      ocscColPtr[col] = nnz;
    }
  }
  ocscColPtr[cols] = nnz;
  int onnz = nnz;
  int* ocscRowInd = (int*)qmalloc(sizeof(int) * onnz, __FUNCTION__, __LINE__);
  QValue* ocscVals = (QValue*)qmalloc(sizeof(QValue) * onnz, __FUNCTION__, __LINE__);
  memcpy(ocscRowInd, cooRowIndex, sizeof(int) * onnz);
  memcpy(ocscVals, cooVal, sizeof(QValue) * onnz);
  CSC csc(ocscVals, ocscRowInd, ocscColPtr, rows, cols, onnz);
  return csc;
}


