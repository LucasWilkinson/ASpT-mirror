#ifndef UTIL_H_
#define UTIL_H_
#include <stdlib.h>
#include <utility>
#include <stdio.h>
#include <vector>
#include <time.h>
#include "tools/macro.h"
using namespace std;

#define MLMCL_PRUNE_A	(0.90) /* pruning parameter */
#define MLMCL_PRUNE_B	(2)	   /* pruning parameter */

QValue computeThreshold(QValue avg, QValue max);
QValue arrayMax(const QValue values[], const int count);
QValue arraySum(const QValue values[], const int count);
pair<QValue, QValue> arrayMaxSum(const QValue values[], const int count);
void arrayInflationR2(const QValue ivalues[], const int count, QValue ovalues[]);
QValue arrayThreshPruneNormalize(const QValue thresh, const int rindices[], const QValue rvalues[],
    int* count, int indices[], QValue values[]);
void arrayOutput(const char* msg, FILE* fp, const int datas[], int len);
void arrayOutput(const char *msg, FILE* fp, const long datas[], int len);
void arrayOutput(const char* msg, FILE* fp, const QValue datas[], int len);
void arrayOutput(const char *msg, FILE* fp, const vector<int> &datas);
void arrayOutput(const char *msg, FILE* fp, const vector<QValue> &datas);
void prefixSumToCounts(const int prefixSum[], const int len, int *counts);
void arrayEqualPartition64(long prefixSum[], const int n, const int nthreads, int ends[]);
void arrayEqualPartition(int prefixSum[], const int n, const int nthreads, int ends[]);
int* randomPermutationVector(const int len);
int* permutationTranspose(const int P[], const int len);
#endif
