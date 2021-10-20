#ifndef PREFIX_SUM_H_
#define PREFIX_SUM_H_
#include <omp.h>
#include <string.h>
#include <algorithm>
#define MAX_THREADS_NUM 250

void ompPrefixSum(int a[], int s[], int n);
void noTileOmpPrefixSum(int a[], int s[], int n);

inline void seqPrefixSum(int a[], int s[], int n) {
  int t0 = a[0];
  int t1;
  s[0] = 0;
  for (int i = 0; i < n; ++i) {
    t1 = a[i + 1];
    s[i + 1] = s[i] + t0;
    t0 = t1;
  }
}
#endif
