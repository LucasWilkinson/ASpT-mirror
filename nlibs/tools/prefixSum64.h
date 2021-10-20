#ifndef PREFIX_SUM64_H_
#define PREFIX_SUM64_H_
#include <omp.h>
#include <string.h>
#include <algorithm>
#define MAX_THREADS_NUM 250

void ompPrefixSum64(long a[], long s[], int n);
void noTileOmpPrefixSum64(long a[], long s[], int n);

inline void seqPrefixSum64(long a[], long s[], int n) {
  long t0 = a[0];
  long t1;
  s[0] = 0;
  for (int i = 0; i < n; ++i) {
    t1 = a[i + 1];
    s[i + 1] = s[i] + t0;
    t0 = t1;
  }
}
#endif
