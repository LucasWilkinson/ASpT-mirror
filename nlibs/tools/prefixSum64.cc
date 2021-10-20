#include "tools/prefixSum64.h"
#include "tools/qmalloc.h"

long pass1_scalar64(long a[], long s[], int n) {
  if (n <= 0) {
    return 0;
  }
  long ret = a[n - 1];
  long t0 = a[0];
  long t1;
  s[0] = 0;
  for (int i = 0; i < n - 1; ++i) {
    t1 = a[i + 1];
    s[i + 1] = s[i] + t0;
    t0 = t1;
  }
  return ret + s[n - 1];
}

void pass2_scalar64(long*s, long offset, const int n) {
  if (n <= 0) {
    return;
  }
//  __assume_aligned(s, 16);
//#pragma ivdep
  for (int i = 0; i < n; ++i) {
    s[i] += offset;
  }
}

void noTileOmpPrefixSum64(long a[], long s[], int n) {
  static long suma[MAX_THREADS_NUM];
  const long ta = a[n - 1];
  const int ithread = omp_get_thread_num();
  const int nthreads = omp_get_num_threads();
  const int tchunk = (n + nthreads - 1) / nthreads;
  const int tstart = tchunk * ithread;
  const int tlen = std::min(tchunk, n - tchunk * ithread);
#pragma omp single
  suma[0] = 0;
#pragma omp barrier
  suma[(ithread + 1)] = pass1_scalar64(&a[tstart], &s[tstart], tlen);
#pragma omp barrier
#pragma omp single
      {
        long tmp = 0;
        for (int i = 0; i < (nthreads); i++) {
          tmp += suma[i];
          suma[i] = tmp;
        }
      }
#pragma omp barrier
      long offset = (suma[ithread]);
      pass2_scalar64(&s[tstart], offset, tlen);
#pragma omp barrier
#pragma omp single
    {
      s[n] = s[n - 1] + ta;
    }
#pragma omp barrier
}

//Called inside omp parallel region
void ompPrefixSum64(long a[], long s[], int n) {
  static long suma[MAX_THREADS_NUM];
  //int *suma;
  const int chunk_size = 1 << 18;
  const int nchunks = n % chunk_size == 0 ? n / chunk_size : n / chunk_size + 1;
//#pragma omp parallel
  {
    const int ithread = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    long offset2 = 0;
    for (int c = 0; c < nchunks; c++) {
      const int start = c * chunk_size;
      const int chunk = (c + 1) * chunk_size < n ? chunk_size : n - c * chunk_size;
      const long ta = a[start + chunk - 1];
      int tchunk = (chunk + nthreads - 1) / nthreads;
      //make aligned with tchunk the times of 4.
      tchunk = (((tchunk + 3) >> 2) << 2);
      const int tstart = start + tchunk * ithread;
      tchunk = std::min(tchunk, chunk - tchunk * ithread);
#pragma omp barrier
      suma[(ithread + 1)] = pass1_scalar64(&a[tstart], &s[tstart], tchunk);
#pragma omp barrier
#pragma omp single
      {
        int tmp = 0;
        for (int i = 0; i < (nthreads); i++) {
          tmp += suma[i];
          suma[i] = tmp;
        }
      }
#pragma omp barrier
      long offset = (suma[ithread] + offset2);
      pass2_scalar64(&s[tstart], offset, tchunk);
#pragma omp barrier
      offset2 = s[start + chunk - 1] + ta;
    }
#pragma omp single
    {
      s[n] = offset2;
    }
  }
}

