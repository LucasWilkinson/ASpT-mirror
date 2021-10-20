#include "qrmcl.h"
#include "tools/ntimer.h"
#include "tools/util.h"
#include "gpus/gpu_csr_kernel.h"
#include "process_args.h"
#include <omp.h>

void mtRmclIter(const int maxIter, const CSR Mgt, CSR &Mt, const int stride, const RunOptions runOptions) {
  CSR newMt;
  QValue tsum = 0.0;
  int nthreads = 8;
#pragma omp parallel
#pragma omp master
  nthreads = omp_get_num_threads();
  printf("nthreads=%d\n", nthreads);

  FILE *fp = NULL;
  static const int cpercents[] = {-30.0, -20.0, -5.0, -0.0, 5.0, 20.0, 30.0, 100.0};
  vector<QValue> percents(cpercents, cpercents + sizeof(cpercents) / sizeof(int));
  if (options.stats) {
    fp = fopen("percent.stats", "w");
    fprintf(fp, "rows %d\n", Mt.rows);
    arrayOutput("percent\t", fp, percents);
  }

  //QValue nowTotal = time_in_mill_now();
  double nowTotal = time_in_mill_now();
  thread_data_t* thread_datas = NULL;
  if (runOptions == SOMP || runOptions == OMP || runOptions == CILK || runOptions == HYB) {
    thread_datas = allocateThreadDatas(nthreads, Mt.cols);
  }
  for (int iter = 0; iter < maxIter; ++iter) {
    //QValue now = time_in_mill_now();
    double now = time_in_mill_now();
#ifdef debugging
    Mgt.output("Mgt iter");
    Mt.output("Mt iter");
#endif

    if (runOptions == SOMP) {
      newMt = Mgt.staticOmpRmclOneStep(Mt, thread_datas, stride);
    } else if (runOptions == HYB) {
      newMt = Mgt.hybridOmpRmclOneStep(Mt, thread_datas, stride);
    } else if (runOptions == SFOMP) {
      newMt = Mgt.staticFairRmclOneStep(Mt, stride);
    } else if (runOptions == OMP) {
      newMt = Mgt.ompRmclOneStep(Mt, thread_datas, stride);
    } else if (runOptions == CILK) {
#ifdef enable_cilk
      newMt = Mgt.cilkRmclOneStep(Mt, thread_datas, stride);
#else
      printf("Configuration config without CILK support\n");
      exit(-1);
#endif
    } else if (runOptions == MKL) {
#ifdef enable_mkl
      newMt = Mgt.mklRmclOneStep(Mt, stride);
#else
      printf("Configuration config without MKL support\n");
      exit(-1);
#endif
    } else {
      printf("Multithreaded RunOptions should be either OMP or CILK\n");
      exit(-1);
    }

    if (options.stats) {
      vector<int> counts = Mt.differsStats(newMt, percents);
      char msg[20];
      sprintf(msg, "%d :\t", iter);
      arrayOutput(msg, fp, counts);
    }

    Mt.dispose();
    Mt = newMt;
    printf("%s with stride %d iter %d done in %lf milliseconds\n",
        runOptionsStr[runOptions], stride, iter, time_in_mill_now() - now);
  }
  if (runOptions == SOMP || runOptions == OMP || runOptions == CILK || runOptions == HYB) {
    freeThreadDatas(thread_datas, nthreads);
  }
  if (options.stats) {
    fclose(fp);
  }
  printf("time pass cpu OMP rmcl iters = %lf\n", time_in_mill_now() - nowTotal);
}

void seqRmclIter(const int maxIter, const CSR Mgt, CSR &Mt) {
  CSR newMt;
  //QValue nowTotal = time_in_mill_now();
  double nowTotal = time_in_mill_now();
  QValue tsum = 0.0;
  for (int iter = 0; iter < maxIter; ++iter) {
  //QValue now = time_in_mill_now();
  double now = time_in_mill_now();
    //newMt = Mgt.omp_spmm(Mt);
    newMt = Mgt.spmm(Mt);
    tsum += time_in_mill_now() - now;
  //printf("time pass total = %lf\n", time_in_mill_now() - now);
    //newMt.output("newMt spmm");
    int pos = 0;
    int i;
    for (i = 0; i < newMt.rows; ++i) {
      int count = newMt.rowCount(i);
      QValue* values = (QValue*)malloc(count * sizeof(QValue));
      arrayInflationR2(newMt.values + newMt.rowPtr[i], count, values);
      QValue rmax = arrayMax(values, count);
      QValue rsum = arraySum(values, count);
      QValue thresh = computeThreshold(rsum / count, rmax);
      int* indices = (int*)malloc(count * sizeof(int));
      QValue nsum = arrayThreshPruneNormalize(thresh, newMt.colInd + newMt.rowPtr[i], values,
          &count, indices, values);
      memcpy(newMt.values + pos, values, count * sizeof(QValue));
      memcpy(newMt.colInd + pos, indices, count * sizeof(int));
      newMt.rowPtr[i] = pos;
      pos += count;
      free(values);
      free(indices);
    }
    newMt.rowPtr[newMt.rows] = pos;
    newMt.nnz = pos;
    Mt.dispose();
    Mt = newMt;
    printf("seq iter %d done in %lf milliseconds\n", iter, time_in_mill_now() - now);
  }
  printf("time pass tsum spmm = %lf\n", tsum);
  printf("time pass cpu seq rmcl iters = %lf\n", time_in_mill_now() - nowTotal);
}

CSR rmclInit(COO &cooAt) {
  //cooAt.output("COO");
  cooAt.addSelfLoopIfNeeded();
  cooAt.makeOrdered();
  //cooA.output("COO");
  CSR At = cooAt.toCSR();
  At.averAndNormRowQValue();
  return At;
}

CSR RMCL(const char iname[], int maxIters, RunOptions runOptions) {
  COO cooAt;
  cooAt.readSNAPFile(iname);
  CSR Mt = rmclInit(cooAt);
  //Mt.output("CSR Mt");
  cooAt.dispose();
  CSR Mgt = Mt.deepCopy();
  if (runOptions == MKL) {
    Mt.toOneBasedCSR();
    Mgt.toOneBasedCSR();
  }
  //Mt.output("CSR Mgt");
  //QValue now = time_in_mill_now();
  double now = time_in_mill_now();
  if (runOptions == GPU) {
#ifdef enable_gpu
    gpuRmclIter(maxIters, Mgt, Mt);
#endif
  } else if (runOptions == SEQ) {
    seqRmclIter(maxIters, Mgt, Mt);
  } else {
    mtRmclIter(maxIters, Mgt, Mt, options.stride, runOptions);
  }
  printf("total time pass with RMCL iters = %lf\n", time_in_mill_now() - now);
  Mgt.dispose();
  if (runOptions == MKL) {
    Mt.toZeroBasedCSR();
  }
  return Mt;
}
