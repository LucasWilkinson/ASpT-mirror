#include "tools/stats.h"

void pushToStats(const long stat, std::vector<int> &stats) {
  for (int i = 0; i < stats.size() - 1; ++i) {
    const long bound = (1l << i);
    if (stat <= bound) {
      ++stats[i];
      return;
    }
  }
  ++stats[stats.size() - 1];
}

void outputStats(const std::vector<int> &stats) {
  int i;
  long long sum = 0;
  for (i = 0; i < stats.size(); ++i) {
    sum += stats[i];
  }
  printf("Total sum = %lld\n", sum);
  for (i = 0; i < stats.size() - 1; ++i) {
    const long bound = (1l << i);
    printf("(%ld -> %ld)\t%d\t%.6f\t\n", bound / 2 + 1, bound, stats[i], (QValue)stats[i] / sum);
  }
  const long bound = (1l << i);
  printf("(%ld -> INF)\t%d\t%.6lf\t\n", bound / 2 + 1, stats[i], (QValue)stats[i] / sum);
}

std::vector<int> flopsStats(const int IA[], const int JA[], const QValue A[], const int nnzA,
    const int IB[], const int JB[], const QValue B[], const int nnzB,
    const int m, const int k, const int n) {
  std::vector<int> stats(30, 0);
  for (int i = 0; i < m; ++i) {
    long row_flops = 0;
    for (int jp = IA[i]; jp < IA[i + 1]; ++jp) {
      int j = JA[jp];
      long Brow_j_nnz = IB[j + 1] - IB[j];
      row_flops += Brow_j_nnz;
    }
    pushToStats(row_flops, stats);
  }
  return stats;
} 

// flops stats for rows and not 
std::vector<int> flopsStatsRows(const int IA[], const int JA[], const QValue A[], const int nnzA,
    const int IB[], const int JB[], const QValue B[], const int nnzB,
    const int m, const int k, const int n) { 
  std::vector<int> stats(13, 0); 
  for (int i = 0; i < m; ++i) {
    long row_flops = 0;
    for (int jp = IA[i]; jp < IA[i + 1]; ++jp) { 
      int j = JA[jp]; 
      long Brow_j_nnz = IB[j + 1] - IB[j]; 
      row_flops += Brow_j_nnz; 
    } 
    pushToStats(row_flops, stats); 
  } 
  return stats; 
} 
