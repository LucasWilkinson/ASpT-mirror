#include "util.h"
#include <algorithm>

QValue computeThreshold(QValue avg, QValue max) {
	QValue ret = MLMCL_PRUNE_A * avg * (1 - MLMCL_PRUNE_B * (max - avg));
	ret = (ret > 1.0e-7) ? ret : 1.0e-7;
	ret = (ret > max) ? max : ret;
	return ret;
}

QValue arrayMax(const QValue values[], const int count) {
  QValue rmax = 0.0;
  for ( int i = 0; i < count; ++i) {
    if (rmax < values[i]) {
      rmax = values[i];
    }
  }
  return rmax;
}

pair<QValue, QValue> arrayMaxSum(const QValue values[], const int count) {
  QValue rmax = 0.0;
  QValue rsum = 0.0;
  for ( int i = 0; i < count; ++i) {
    if (rmax < values[i]) {
      rmax = values[i];
    }
    rsum += values[i];
  }
  return make_pair(rmax, rsum);
}

QValue arraySum(const QValue *restrict values, const int count) {
  QValue rsum = 0.0;
  for (int i = 0; i < count; ++i) {
    rsum += values[i];
  }
  return rsum;
}

void arrayInflationR2(const QValue *restrict ivalues, const int count, QValue *restrict ovalues) {
  for (int i = 0; i < count; ++i) {
    ovalues[i] = ivalues[i] * ivalues[i];
  }
}

QValue arrayThreshPruneNormalize(const QValue thresh, const int rindices[], const QValue rvalues[],
    int* count, int indices[], QValue values[]) {
	//int* indicesToRetain = (int*)malloc(sizeof(int) * (*count));
	int i, j;
	QValue sum = 0;
	for (i = 0, j = 0; i < *count; ++i) {
		if (values[i] >= thresh) {
			sum += rvalues[i];
			//indicesToRetain[j++] = i;
      indices[j] = rindices[i];
      values[j++] = rvalues[i];
		}
	}
  //normalize
	for (i = 0; i < j; ++i) {
		//indices[i] = rindices[indicesToRetain[i]];
		//values[i] = rvalues[indicesToRetain[i]] / sum;
		values[i] = values[i] / sum;
	}
	*count = j;
	//free(indicesToRetain);
	return sum;
}

void arrayOutput(const char *msg, FILE* fp, const QValue datas[], int len) {
  fprintf(fp, "%s", msg);
  for (int i = 0; i < len; ++i) {
    fprintf(fp, "%e ", datas[i]);
  }
  fprintf(fp, "\n");
  fflush(fp);
}

void arrayOutput(const char *msg, FILE* fp, const int datas[], int len) {
  fprintf(fp, "%s", msg);
  for (int i = 0; i < len; ++i) {
    fprintf(fp, "%d ", datas[i]);
  }
  fprintf(fp, "\n");
  fflush(fp);
}

void arrayOutput(const char *msg, FILE* fp, const long datas[], int len) {
  fprintf(fp, "%s", msg);
  for (int i = 0; i < len; ++i) {
    fprintf(fp, "%ld ", datas[i]);
  }
  fprintf(fp, "\n");
  fflush(fp);
}

void arrayOutput(const char *msg, FILE* fp, const vector<int> &datas) {
  fprintf(fp, "%s", msg);
  for (int i = 0; i < datas.size(); ++i) {
    fprintf(fp, "%d ", datas[i]);
  }
  fprintf(fp, "\n");
  fflush(fp);
}

void arrayOutput(const char *msg, FILE* fp, const vector<QValue> &datas) {
  fprintf(fp, "%s", msg);
  for (int i = 0; i < datas.size(); ++i) {
    fprintf(fp, "%lf ", datas[i]);
  }
  fprintf(fp, "\n");
  fflush(fp);
}

//len is the length of counts array
void prefixSumToCounts(const int prefixSum[], const int len, int *counts) {
  for (int i = 0; i < len; ++i) {
    counts[i] = prefixSum[i + 1] - prefixSum[i];
  }
}

void arrayEqualPartition64(long prefixSum[], const int n, const int nthreads, int ends[]) {
  const long chunk_size = (prefixSum[n] + nthreads - 1) / nthreads;
  ends[0] = 0;
  for (int i = 0, now = 0; i < nthreads - 1; ++i) {
    const long target = std::min((i + 1) * chunk_size, prefixSum[n]);
    long* begin = prefixSum + now;
    long* upper = std::upper_bound(begin, prefixSum + n + 1, target);
    ends[i + 1] = std::max((int)(upper - prefixSum - 1), now + 1);
    ends[i + 1] = std::min(ends[i + 1], n);
    now = ends[i + 1];
  }
  ends[nthreads] = n;
}

void arrayEqualPartition(int prefixSum[], const int n, const int nthreads, int ends[]) {
  const int chunk_size = (prefixSum[n] + nthreads - 1) / nthreads;
  ends[0] = 0;
  for (int i = 0, now = 0; i < nthreads - 1; ++i) {
    const int target = std::min((i + 1) * chunk_size, prefixSum[n]);
    int* begin = prefixSum + now;
    int* upper = std::upper_bound(begin, prefixSum + n + 1, target);
    ends[i + 1] = std::max((int)(upper - prefixSum - 1), now + 1);
    ends[i + 1] = std::min(ends[i + 1], n);
    now = ends[i + 1];
  }
  ends[nthreads] = n;
}

int* randomPermutationVector(const int len) {
  srand(time(NULL));
  int* P = (int*)malloc(len * sizeof(int));
  for (int i = 0; i < len; ++i) {
    int pos = rand() % (i + 1);
    P[i] = P[pos];
    P[pos] = i;
  }
  return P;
}

int* permutationTranspose(const int P[], const int len) {
  int* Pt = (int*)malloc(len * sizeof(int));
  for (int i = 0; i < len; ++i) {
    Pt[P[i]] = i;
  }
  return Pt;
}
