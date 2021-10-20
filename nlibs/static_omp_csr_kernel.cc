#include <omp.h>
#include <iostream>
#include "cpu_csr_kernel.h"
#include "tools/ntimer.h"
#include "tools/util.h"
#include "tools/prefixSum.h"
using namespace std;

long spmmFootPrints(const int IA[], const int JA[],
    const int IB[], const int IC[],
    const int m,
    long *footPrintSum) {
  long footPrints = 0;
  footPrintSum[0] = 0;
  for (int i = 0; i < m; ++i) {
    long row_flops = 0;
    for (int jp = IA[i]; jp < IA[i + 1]; ++jp) {
      int j = JA[jp];
      long Brow_j_nnz = IB[j + 1] - IB[j];
      row_flops += Brow_j_nnz;
    }
    footPrints += row_flops + IC[i + 1] - IC[i] + 1;
    footPrintSum[i + 1] = footPrints;
  }
  return footPrints;
}


inline int compute_C_row_memory_Managed_2(const int i, const int m, const int IA[], const int JA[], const QValue A[], 
  const int IB[], const int JB[], const QValue B[],
  int* iJC_initial_chunk, int* iJC_later_chunk, int* &iJC_extra_chunk, 
  QValue* iC_initial_chunk, QValue* iC_later_chunk, QValue* &iC_extra_chunk,
  const int initial_chunk_size, const int later_chunk_size, const int extra_chunk_size,
  QValue* xb[], int& chunk_lock, int chunk_factor, int thread_chunk_index[], bool& extra_chunk_allocated) {
  
  if (IA[i] == IA[i + 1]) {
    return 0;
  }

  //count: track index in the current chunk,  total_count: tracks total no. of nnZ's for this row
  int count = -1, total_count = 0;
  bool initial_chunk=true, later_chunk=false;
  int thread_id = omp_get_thread_num();

  int* iJC = iJC_initial_chunk + i*initial_chunk_size;
  QValue* iC = iC_initial_chunk + i*initial_chunk_size;
  
   for (int ap = IA[i]; ap < IA[i + 1]; ++ap) {
    int v = JA[ap];
    QValue aVal = A[ap];
    for (int bp = IB[v]; bp < IB[v+1]; ++bp) {
      int k = JB[bp];
      if(xb[k] == NULL) {
        count++;
        if(initial_chunk) {
          if(count==initial_chunk_size-1) {
                                    
            total_count += count; 
            count = 0;
            initial_chunk=false;
            
            int chunk_index = ++thread_chunk_index[thread_id];
            //checking if the current thread has any chunks left in later_chunk memory pool
            if(chunk_index < chunk_factor*(thread_id+1)) {
              later_chunk=true;
            }
            if(later_chunk) {
            // setting the last index of current chunk to the index of the next chunk
            iJC[initial_chunk_size-1] = chunk_index;

            // updating iJC, iC to the new chunk where values have to be updated
            iJC = iJC_later_chunk + chunk_index*later_chunk_size;
            iC = iC_later_chunk + chunk_index*later_chunk_size; 

            }
            else{
              if(extra_chunk_allocated) {

                #pragma omp atomic capture
                chunk_index = ++chunk_lock;

              }
              else {
                #pragma omp critical 
                {
                  if(!extra_chunk_allocated) {
                      iJC_extra_chunk = (int*)malloc(extra_chunk_size * 2 * m * sizeof(int));
                      iC_extra_chunk = (QValue*)malloc(extra_chunk_size * 2 * m * sizeof(QValue));
                      chunk_lock = 0;
                      chunk_index = 0;
                      extra_chunk_allocated = true;
                    }
                    else {
                      #pragma omp atomic capture
                      chunk_index = ++chunk_lock;
                    }
                }
              }
              // setting the last index of current chunk to the index of the next chunk, setting it to -ve value to indicate extra_chunk
              iJC[initial_chunk_size-1] = (chunk_index+1) * -1;

              // updating iJC, iC to the new chunk where values have to be updated
              iJC = iJC_extra_chunk + chunk_index*extra_chunk_size;
              iC = iC_extra_chunk + chunk_index*extra_chunk_size; 

            }
          } 
          iJC[count] = k;
          iC[count] = aVal * B[bp];
          xb[k] = &iC[count];
        }
        else if(later_chunk) {
          if(count==later_chunk_size-1) {

            total_count += count; 
            count = 0;

            int chunk_index = ++thread_chunk_index[thread_id];

            //checking if the current thread has any chunks left in later_chunk memory
            if(chunk_index >= chunk_factor*(thread_id+1)) {
              later_chunk=false;
            }

            if(later_chunk) {
            // setting the last index of current chunk to the index of the next chunk
            iJC[later_chunk_size-1] = chunk_index;

            // updating iJC, iC to the new chunk where values have to be updated
            iJC = iJC_later_chunk + chunk_index*later_chunk_size;
            iC = iC_later_chunk + chunk_index*later_chunk_size; 

            }
            else{
              if(extra_chunk_allocated) {

                #pragma omp atomic capture
                chunk_index = ++chunk_lock;

              }
              else {
                #pragma omp critical
                {
                  if(!extra_chunk_allocated) {
                      iJC_extra_chunk = (int*)malloc(extra_chunk_size * 2 * m * sizeof(int));
                      iC_extra_chunk = (QValue*)malloc(extra_chunk_size * 2 * m * sizeof(QValue));
                      chunk_lock = 0;
                      chunk_index = 0;
                      extra_chunk_allocated = true;
                    }
                    else {
                      #pragma omp atomic capture
                      chunk_index = ++chunk_lock;
                    }
                }
              }
              // setting the last index of current chunk to the index of the next chunk, setting it to -ve value to indicate extra_chunk
              iJC[later_chunk_size-1] = (chunk_index+1) * -1;

              // updating iJC, iC to the new chunk where values have to be updated
              iJC = iJC_extra_chunk + chunk_index*extra_chunk_size;
              iC = iC_extra_chunk + chunk_index*extra_chunk_size; 
            }   
          } 
          iJC[count] = k;
          iC[count] = aVal * B[bp];
          xb[k] = &iC[count];
        }
        else {
          if(count==extra_chunk_size-1) {
            total_count += count; 
            count = 0;  

            int chunk_index;
            #pragma omp atomic capture
            chunk_index = ++chunk_lock;
            // setting the last index of current chunk to the index of the next chunk
            iJC[extra_chunk_size-1] = chunk_index;

            // updating iJC, iC to the new chunk where values have to be updated
            iJC = iJC_extra_chunk + chunk_index*extra_chunk_size;
            iC = iC_extra_chunk + chunk_index*extra_chunk_size; 
          } 
          iJC[count] = k;
          iC[count] = aVal * B[bp];
          xb[k] = &iC[count];
        }

      }
      else {
        *xb[k] += aVal * B[bp];
      }
    }
  }
  
  total_count += (count + 1);

  count=0; initial_chunk=true; later_chunk=false;
  iJC = iJC_initial_chunk + i*initial_chunk_size;
  

  // re-setting the scatter-vector to NULL
  for(int jp = 0; jp < total_count; ++jp, count++) {
    int j;
    if(initial_chunk) {
      if(count == initial_chunk_size-1) {
        initial_chunk=false;
        count=0;

        int chunk_index = iJC[initial_chunk_size-1];
        if(chunk_index >= 0) {
          later_chunk=true;
          iJC = iJC_later_chunk + chunk_index*later_chunk_size;
        }
        else {
          chunk_index= (chunk_index * (-1)) - 1;
          iJC = iJC_extra_chunk + chunk_index*extra_chunk_size;
        }
      }
      j=iJC[count];
    } 
    else if(later_chunk) {
      if(count == later_chunk_size-1) {
        count = 0;
        
        int chunk_index = iJC[later_chunk_size-1];
        if(chunk_index >= 0) {
          iJC = iJC_later_chunk + chunk_index*later_chunk_size;
        }
        else {
          later_chunk=false;
          chunk_index= (chunk_index * (-1)) - 1;
          iJC = iJC_extra_chunk + chunk_index*extra_chunk_size;
        }
      }
      j=iJC[count];
    }
    else {
      if(count == extra_chunk_size-1) {
        count=0;
        int chunk_index = iJC[extra_chunk_size-1];
        //printf("chunk_index:%d\n",chunk_index);
        iJC = iJC_extra_chunk + chunk_index*extra_chunk_size;
      }
      j=iJC[count];
    } 
    
    xb[j] = NULL;
  }

  //return nnzC for this row
  return total_count;
}


/*
 * dynamic_omp_CSR_IC_nnzC_computeC_memory_Managed -reminder: this function must be called in #pragma omp parallel regions
 * to parallelly execution.
 * */
void dynamic_omp_CSR_IC_nnzC_compute_C_memory_Managed_2(const int IA[], const int JA[], const QValue A[],
    const int IB[], const int JB[], const QValue B[],
    const int m, const int n, const thread_data_t& thread_data,
    int* IC, int& nnzC, const int stride, int* iJC_initial_chunk, int* iJC_later_chunk, int* &iJC_extra_chunk,
    QValue* iC_initial_chunk, QValue* iC_later_chunk, QValue* &iC_extra_chunk, 
    const int initial_chunk_size, const int later_chunk_size, const int extra_chunk_size, 
    int& chunk_lock, int chunk_factor, int thread_chunk_index[], bool& extra_chunk_allocated) {

  //initialize  thread_data
  QValue **xb = (QValue**)thread_data.xbp;
  memset(xb, NULL, n * sizeof(QValue*));

// #pragma omp for schedule(dynamic)
//   for (int it = 0; it < m; it += stride) {
//     int up = it + stride < m ? it + stride : m;
//     for (int i = it; i < up; ++i) {
//       IC[i] = compute_C_row_memory_Managed(i, IA, JA, A, IB, JB, B, iJC_initial_chunk, iJC_later_chunk, 
//         iC_initial_chunk, iC_later_chunk, initial_chunk_size, later_chunk_size, chunk_lock, xb);
//     }
//   }

#pragma omp for schedule(dynamic, stride)
  for (int i = 0; i < m; i++) {
    IC[i] = compute_C_row_memory_Managed_2(i, m, IA, JA, A, IB, JB, B, iJC_initial_chunk, iJC_later_chunk, iJC_extra_chunk,
        iC_initial_chunk, iC_later_chunk, iC_extra_chunk, initial_chunk_size, later_chunk_size, extra_chunk_size, xb,
        chunk_lock, chunk_factor, thread_chunk_index, extra_chunk_allocated);
  }

#pragma omp barrier
  noTileOmpPrefixSum(IC, IC, m);
#pragma omp single
  {
    nnzC = IC[m];
  }
}



void copy_C_row_memory_Managed_2(const int i, const int nnz_C_row, int JC[], QValue C[],
  int iJC_initial_chunk[], int iJC_later_chunk[], int iJC_extra_chunk[], 
  QValue iC_initial_chunk[], QValue iC_later_chunk[], QValue iC_extra_chunk[],
  const int initial_chunk_size, const int later_chunk_size, const int extra_chunk_size) {
  
  if (nnz_C_row == 0) {
    return;
  }
  
  //count: track index in the current chunk,  total_count: total no. of nnZ's for this row
  int count = 0, total_count = nnz_C_row;
  bool initial_chunk=true, later_chunk=false;

  int* iJC = iJC_initial_chunk + i*initial_chunk_size;
  QValue* iC = iC_initial_chunk + i*initial_chunk_size;
    
  for(int j = 0; j < total_count; j++, count++) {
    
    if(initial_chunk) {
      if(count == initial_chunk_size-1) {
        initial_chunk=false;
        count=0;

        int chunk_index = iJC[initial_chunk_size-1];
        if(chunk_index >= 0) {
          later_chunk=true;
          iJC = iJC_later_chunk + chunk_index*later_chunk_size;
          iC = iC_later_chunk + chunk_index*later_chunk_size;
        }
        else {
          chunk_index= (chunk_index * (-1)) - 1;
          iJC = iJC_extra_chunk + chunk_index*extra_chunk_size;
          iC = iC_extra_chunk + chunk_index*extra_chunk_size;
        }
          
      }
      JC[j] = iJC[count];
      C[j] = iC[count];
    }
    else if(later_chunk) {
      if(count == later_chunk_size-1) {
        count = 0;

        int chunk_index = iJC[later_chunk_size-1];
        if(chunk_index >= 0) {
          iJC = iJC_later_chunk + chunk_index*later_chunk_size;
          iC = iC_later_chunk + chunk_index*later_chunk_size;
        }
        else {
          later_chunk=false;
          chunk_index= (chunk_index * (-1)) - 1;
          iJC = iJC_extra_chunk + chunk_index*extra_chunk_size;
          iC = iC_extra_chunk + chunk_index*extra_chunk_size;
        }
               
      }
      JC[j] = iJC[count];
      C[j] = iC[count];

    } 
    else {
      if(count == extra_chunk_size-1) {
        count=0;
        int chunk_index = iJC[extra_chunk_size-1];
        iJC = iJC_extra_chunk + chunk_index*extra_chunk_size;
        iC = iC_extra_chunk + chunk_index*extra_chunk_size;
      }
      JC[j] = iJC[count];
      C[j] = iC[count];
    } 

  }

}



void dynamic_omp_CSR_copy_C_memory_Managed_2(const int m, const int n, int* IC, int* JC, QValue* C, 
  int& nnzC, const int stride, int* iJC_initial_chunk, int* iJC_later_chunk, int* iJC_extra_chunk,
  QValue* iC_initial_chunk, QValue* iC_later_chunk, QValue* iC_extra_chunk,
  const int initial_chunk_size, const int later_chunk_size, const int extra_chunk_size) {
 
 // #pragma omp for schedule(dynamic)
 //  for (int it = 0; it < m; it += stride) {
 //    int up = it + stride < m ? it + stride : m;
 //    for (int i = it; i < up; ++i) {
 //      copy_C_row_memory_Managed(i, IC[i+1] - IC[i], JC + IC[i], C + IC[i], iJC_initial_chunk, iJC_later_chunk, 
 //        iC_initial_chunk, iC_later_chunk, initial_chunk_size, later_chunk_size);
 //    }
 //  }

 #pragma omp for schedule(dynamic, stride)
  for (int i = 0; i < m; i++) {
      copy_C_row_memory_Managed_2(i, IC[i+1] - IC[i], JC + IC[i], C + IC[i], iJC_initial_chunk, iJC_later_chunk, iJC_extra_chunk, 
          iC_initial_chunk, iC_later_chunk, iC_extra_chunk, initial_chunk_size, later_chunk_size, extra_chunk_size);
  }

}


inline int compute_C_row_memory_Managed(const int i, const int IA[], const int JA[], const QValue A[], 
  const int IB[], const int JB[], const QValue B[],
  int iJC_initial_chunk[], int iJC_later_chunk[], QValue iC_initial_chunk[], QValue iC_later_chunk[], 
  const int initial_chunk_size, const int later_chunk_size, int& chunk_lock, QValue* xb[]) {
  
  if (IA[i] == IA[i + 1]) {
    return 0;
  }

  //count: track index in the current chunk,  total_count: tracks total no. of nnZ's for this row
  int count = -1, total_count = 0;
  bool initial_chunk=true;

  int* iJC = iJC_initial_chunk + i*initial_chunk_size;
  QValue* iC = iC_initial_chunk + i*initial_chunk_size;
  
   for (int ap = IA[i]; ap < IA[i + 1]; ++ap) {
    int v = JA[ap];
    QValue aVal = A[ap];
    for (int bp = IB[v]; bp < IB[v+1]; ++bp) {
      int k = JB[bp];
      if(xb[k] == NULL) {
        count++;
        if(initial_chunk) {
          if(count==initial_chunk_size-1) {
            int chunk_index;
            #pragma omp atomic capture
              chunk_index = ++chunk_lock;
            // setting the last index of current chunk to the index of the next chunk
            iJC[initial_chunk_size-1] = chunk_index;

            // updating iJC, iC to the new chunk where values have to be updated
            iJC = iJC_later_chunk + chunk_index*later_chunk_size;
            iC = iC_later_chunk + chunk_index*later_chunk_size; 

            total_count += count; 
            count = 0;
            initial_chunk=false;

          } 
          iJC[count] = k;
          iC[count] = aVal * B[bp];
          xb[k] = &iC[count];
        }
        else {
          if(count==later_chunk_size-1) {
            int chunk_index;
            #pragma omp atomic capture
              chunk_index = ++chunk_lock;
            // setting the last index of current chunk to the index of the next chunk
            iJC[later_chunk_size-1] = chunk_index;

            // updating iJC, iC to the new chunk where values have to be updated
            iJC = iJC_later_chunk + chunk_index*later_chunk_size;
            iC = iC_later_chunk + chunk_index*later_chunk_size; 

            total_count += count; 
            count = 0;    
          } 
          iJC[count] = k;
          iC[count] = aVal * B[bp];
          xb[k] = &iC[count];
        }
      }
      else {
        *xb[k] += aVal * B[bp];
      }
    }
  }
  
  total_count += (count + 1);

  count=0; initial_chunk=true;
  iJC = iJC_initial_chunk + i*initial_chunk_size;
  

  // re-setting the scatter-vector to NULL
  for(int jp = 0; jp < total_count; ++jp, count++) {
    int j;
    if(initial_chunk) {
      if(count == initial_chunk_size-1) {
        int chunk_index = iJC[initial_chunk_size-1];
        iJC = iJC_later_chunk + chunk_index*later_chunk_size;
        count = 0;
        initial_chunk=false;
      }
      j=iJC[count];
    } 
    else {
      if(count == later_chunk_size-1) {
        int chunk_index = iJC[later_chunk_size-1];
        iJC = iJC_later_chunk + chunk_index*later_chunk_size;
        count = 0;
      }
      j=iJC[count];
    } 

    xb[j] = NULL;
  }

  //return nnzC for this row
  return total_count;
}





/*
 * dynamic_omp_CSR_IC_nnzC_computeC_memory_Managed -reminder: this function must be called in #pragma omp parallel regions
 * to parallelly execution.
 * */
void dynamic_omp_CSR_IC_nnzC_compute_C_memory_Managed(const int IA[], const int JA[], const QValue A[],
    const int IB[], const int JB[], const QValue B[],
    const int m, const int n, const thread_data_t& thread_data,
    int* IC, int& nnzC, const int stride, int* iJC_initial_chunk, int* iJC_later_chunk, 
    QValue* iC_initial_chunk, QValue* iC_later_chunk, const int initial_chunk_size, const int later_chunk_size, int& chunk_lock) {

  //initialize  thread_data
  QValue **xb = (QValue**)thread_data.xbp;
  memset(xb, NULL, n * sizeof(QValue*));

// #pragma omp for schedule(dynamic)
//   for (int it = 0; it < m; it += stride) {
//     int up = it + stride < m ? it + stride : m;
//     for (int i = it; i < up; ++i) {
//       IC[i] = compute_C_row_memory_Managed(i, IA, JA, A, IB, JB, B, iJC_initial_chunk, iJC_later_chunk, 
//         iC_initial_chunk, iC_later_chunk, initial_chunk_size, later_chunk_size, chunk_lock, xb);
//     }
//   }

#pragma omp for schedule(dynamic, stride)
  for (int i = 0; i < m; i++) {
    IC[i] = compute_C_row_memory_Managed(i, IA, JA, A, IB, JB, B, iJC_initial_chunk, iJC_later_chunk, 
        iC_initial_chunk, iC_later_chunk, initial_chunk_size, later_chunk_size, chunk_lock, xb);
  }

#pragma omp barrier
  noTileOmpPrefixSum(IC, IC, m);
#pragma omp single
  {
    nnzC = IC[m];
  }
}


void copy_C_row_memory_Managed(const int i, const int nnz_C_row, int JC[], QValue C[],
  int iJC_initial_chunk[], int iJC_later_chunk[], QValue iC_initial_chunk[], QValue iC_later_chunk[], 
  const int initial_chunk_size, const int later_chunk_size) {
  
  if (nnz_C_row == 0) {
    return;
  }
  
  //count: track index in the current chunk,  total_count: total no. of nnZ's for this row
  int count = 0, total_count = nnz_C_row;
  bool initial_chunk=true;

  int* iJC = iJC_initial_chunk + i*initial_chunk_size;
  QValue* iC = iC_initial_chunk + i*initial_chunk_size;
    
  for(int j = 0; j < total_count; j++, count++) {
    
    if(initial_chunk) {
      if(count == initial_chunk_size-1) {
        int chunk_index = iJC[initial_chunk_size-1];
        iJC = iJC_later_chunk + chunk_index*later_chunk_size;
        iC = iC_later_chunk + chunk_index*later_chunk_size;
        count = 0;
        initial_chunk=false;
      }
      JC[j] = iJC[count];
      C[j] = iC[count];
    } 
    else {
      if(count == later_chunk_size-1) {
        int chunk_index = iJC[later_chunk_size-1];
        iJC = iJC_later_chunk + chunk_index*later_chunk_size;
        iC = iC_later_chunk + chunk_index*later_chunk_size;
        count = 0;
      }
      JC[j] = iJC[count];
      C[j] = iC[count];
    } 

  }

}


void dynamic_omp_CSR_copy_C_memory_Managed(const int m, const int n, int* IC, int* JC, QValue* C, 
  int& nnzC, const int stride, int* iJC_initial_chunk, int* iJC_later_chunk, 
  QValue* iC_initial_chunk, QValue* iC_later_chunk, const int initial_chunk_size, const int later_chunk_size) {
 
 // #pragma omp for schedule(dynamic)
 //  for (int it = 0; it < m; it += stride) {
 //    int up = it + stride < m ? it + stride : m;
 //    for (int i = it; i < up; ++i) {
 //      copy_C_row_memory_Managed(i, IC[i+1] - IC[i], JC + IC[i], C + IC[i], iJC_initial_chunk, iJC_later_chunk, 
 //        iC_initial_chunk, iC_later_chunk, initial_chunk_size, later_chunk_size);
 //    }
 //  }

 #pragma omp for schedule(dynamic, stride)
  for (int i = 0; i < m; i++) {
      copy_C_row_memory_Managed(i, IC[i+1] - IC[i], JC + IC[i], C + IC[i], iJC_initial_chunk, iJC_later_chunk, 
          iC_initial_chunk, iC_later_chunk, initial_chunk_size, later_chunk_size);
  }

}


void static_omp_CSR_SpMM_memory_Managed(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride) {

  // VT profiling
  double nowt = time_in_mill_now();
  double nowl;

  IC = (int*)malloc((m + 1) * sizeof(int));
  
  int chunk_lock = -1;
  int initial_chunk_size = 256, later_chunk_size = 256, extra_chunk_size = 256;
  int *iJC_initial_chunk, *iJC_later_chunk, *iJC_extra_chunk;
  QValue *iC_initial_chunk, *iC_later_chunk, *iC_extra_chunk;

  //double now = time_in_mill_now();
    
  iJC_initial_chunk = (int*)malloc(initial_chunk_size * m * sizeof(int));
  iJC_later_chunk = (int*)malloc(later_chunk_size * 3 * m * sizeof(int));

  iC_initial_chunk = (QValue*)malloc(initial_chunk_size * m * sizeof(QValue));
  iC_later_chunk = (QValue*)malloc(later_chunk_size * 3 * m * sizeof(QValue));

  //double now2 = time_in_mill_now() - now;
  //printf("malloc() Time:%lf\n",now2);

  int chunk_factor;
  bool extra_chunk_allocated = false;
  int thread_chunk_index[MAX_THREADS_NUM];


#pragma omp parallel firstprivate(stride)
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();

     // dynamic_omp_CSR_IC_nnzC_compute_C_memory_Managed(IA, JA, A, IB, JB, B, m, n, thread_datas[tid], IC, nnzC, stride, iJC_initial_chunk, iJC_later_chunk, iC_initial_chunk, iC_later_chunk, initial_chunk_size, later_chunk_size, chunk_lock);

      #pragma omp single
      {
        //printf("nthreads:%d\n",nthreads);
        chunk_factor = (m*3)/nthreads;
        for(int i=0; i<nthreads; i++) {
          thread_chunk_index[i] = chunk_factor*i;
        }
      }
    
       dynamic_omp_CSR_IC_nnzC_compute_C_memory_Managed_2(IA, JA, A, IB, JB, B, m, n, thread_datas[tid], IC, nnzC, stride, iJC_initial_chunk, iJC_later_chunk, iJC_extra_chunk, iC_initial_chunk, iC_later_chunk, iC_extra_chunk, initial_chunk_size, later_chunk_size, extra_chunk_size, chunk_lock, chunk_factor, thread_chunk_index, extra_chunk_allocated);

  #pragma omp barrier
  #pragma omp single
    { 
      //VT profiling
      printf("Time passed for phase 1 mem managed is %lf milliseconds\n", time_in_mill_now() - nowt);
      nowl = time_in_mill_now();
   
      //printf("initial_chunk_size:%d\t later_chunk_size:%d\t chunk_lock:%d\t nthreads:%d \n", initial_chunk_size, later_chunk_size, chunk_lock, nthreads);
      JC = (int*)malloc(sizeof(int) * nnzC);
      C = (QValue*)malloc(sizeof(QValue) * nnzC);
    }

     // dynamic_omp_CSR_copy_C_memory_Managed(m, n, IC, JC, C, nnzC, stride, iJC_initial_chunk, iJC_later_chunk, iC_initial_chunk, iC_later_chunk, initial_chunk_size, later_chunk_size);

    dynamic_omp_CSR_copy_C_memory_Managed_2(m, n, IC, JC, C, nnzC, stride, iJC_initial_chunk, iJC_later_chunk, iJC_extra_chunk, 
      iC_initial_chunk, iC_later_chunk, iC_extra_chunk, initial_chunk_size, later_chunk_size, extra_chunk_size);
  
  }
  
  free(iJC_initial_chunk);
  free(iJC_later_chunk);
  free(iC_initial_chunk);
  free(iC_later_chunk);
  if(extra_chunk_allocated) {
    free(iJC_extra_chunk);
    free(iC_extra_chunk);
  }
  //VT profiling
   printf("Time passed for phase 2 mem managed is %lf milliseconds\n", time_in_mill_now() - nowl);
   printf("Time passed for total mem managed is %lf milliseconds\n", time_in_mill_now() - nowt);

}


void static_omp_CSR_SpMM_memory_Managed(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const int stride) {

    int nthreads = 8;
#pragma omp parallel
#pragma omp master
    nthreads = omp_get_num_threads();
    
    thread_data_t* thread_datas = allocateThreadDatas_memory_Managed(nthreads, n);
    static_omp_CSR_SpMM_memory_Managed(IA, JA, A, nnzA,
        IB, JB, B, nnzB,
        IC, JC, C, nnzC,
        m, k, n, thread_datas, stride);
    freeThreadDatas_memory_Managed(thread_datas, nthreads);
}


inline int footPrintsCrowiCount(const int i, const int IA[], const int JA[], const int IB[], const int JB[], int iJC[], bool xb[], int &footPrints) {
  if (IA[i] == IA[i + 1]) {
    return 0;
  }
  int count = -1;
  int vp = IA[i];
  int v = JA[vp];
  footPrints = 0;
  for (int kp = IB[v]; kp < IB[v+1]; ++kp) {
    int k = JB[kp];
    iJC[++count] = k;
    xb[k] = true;
  }
  footPrints += IB[v + 1] - IB[v];
  for (int vp = IA[i] + 1; vp < IA[i + 1]; ++vp) {
    int v = JA[vp];
    for (int kp = IB[v]; kp < IB[v+1]; ++kp) {
      int k = JB[kp];
      if(xb[k] == false) {
        iJC[++count] = k;
        xb[k] = true;
      }
    }
    footPrints += IB[v + 1] - IB[v];
  }
  ++count;
  for(int jp = 0; jp < count; ++jp) {
    int j = iJC[jp];
    xb[j] = false;
  }
  footPrints += count + 32 + (IA[i + 1] - IA[i]);
  //footPrints += count + 1;
  footPrints >>= 1; // The way to remove integer overflow in later prefix sum.
  return count;
}

/*
 * dynamic_omp_CSR_IC_nnzC_footprints reminder: this function must be called in #pragma omp parallel regions
 * to parallelly execution.
 * */
void dynamic_omp_CSR_IC_nnzC_footprints(const int IA[], const int JA[],
    const int IB[], const int JB[],
    const int m, const int n, const thread_data_t& thread_data,
    int* IC, int& nnzC, int* footPrints, const int stride) {
  int *iJC = (int*)thread_data.index;
  bool *xb = thread_data.xb;
#ifdef profiling
    QValue xnow = time_in_mill_now();
#endif
  memset(xb, 0, n);
#ifdef profiling
  printf("Time passed for thread %d memset xb with %lf milliseconds\n", omp_get_thread_num(), time_in_mill_now() - xnow);
#endif
#pragma omp for schedule(dynamic)
  for (int it = 0; it < m; it += stride) {
    int up = it + stride < m ? it + stride : m;
    for (int i = it; i < up; ++i) {
      IC[i] = footPrintsCrowiCount(i, IA, JA, IB, JB, iJC, xb, footPrints[i]);
    }
  }
#pragma omp barrier
  noTileOmpPrefixSum(IC, IC, m);
  noTileOmpPrefixSum(footPrints, footPrints, m);
#pragma omp single
  {
    nnzC = IC[m];
  }
}

//int indexRowId = -1;
void static_omp_CSR_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride) {
  // VT profiling
  double nowt = time_in_mill_now();
  double nowl;
#ifdef profiling
  QValue now = time_in_mill_now();
#endif
  IC = (int*)malloc((m + 1) * sizeof(int));
  int* footPrints = (int*)malloc((m + 1) * sizeof(int));
#ifdef profiling
  printf("Time passed for malloc IC and footprints with %lf milliseconds\n", time_in_mill_now() - now);
  now = time_in_mill_now();
#endif
  static int ends[MAX_THREADS_NUM];
#pragma omp parallel firstprivate(stride)
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
#ifdef profiling
    QValue now = time_in_mill_now();
#endif
    dynamic_omp_CSR_IC_nnzC_footprints(IA, JA, IB, JB, m, n, thread_datas[tid], IC, nnzC, footPrints, stride);
#ifdef profiling
    printf("Time passed for thread %d footprints nnzC with %lf milliseconds\n", tid, time_in_mill_now() - now);
#endif
#pragma omp barrier
#pragma omp single
    {
 

#ifdef profiling
      QValue now = time_in_mill_now();
#endif
      //spmmFootPrints(IA, JA, IB, IC, m, footPrints);
      // VT commented below
      //printf("nthreads:%d\n", nthreads);
      //arrayEqualPartition(footPrints, m, nthreads, ends);
#ifdef profiling
      std::cout << "time passed for just partition " << time_in_mill_now() - now << std::endl;
      arrayOutput("ends partitions ", stdout, ends, nthreads + 1);
      printf("Footprints partitions\n");
      for (int i = 0; i < nthreads; ++i) {
        printf("%d ", footPrints[ends[i + 1]] - footPrints[ends[i]]);
      }
      printf("\n");
      //std::cout << "time passed for footPrints and partition " << time_in_mill_now() - now << std::endl;
#endif
      //VT profiling     
      printf("Time passed for phase 1 somp  is %lf milliseconds\n", time_in_mill_now() - nowt);
      nowl = time_in_mill_now();
    }
#pragma omp master
    {
#ifdef profiling
      QValue mnow = time_in_mill_now();
#endif
      JC = (int*)malloc(sizeof(int) * nnzC);
      C = (QValue*)malloc(sizeof(QValue) * nnzC);
#ifdef profiling
      printf("time passed for malloc JC and C in main thread with %lf milliseconds\n", time_in_mill_now() - mnow);
      now = time_in_mill_now();
#endif
    }
    QValue *x = thread_datas[tid].x;
    int *index = thread_datas[tid].index;
#ifdef profiling
      QValue inow = time_in_mill_now();
#endif
    memset(index, -1, n * sizeof(int));
#ifdef profiling
    printf("Time passed for thread %d memset index with %lf milliseconds\n", tid, time_in_mill_now() - inow);
#endif
#pragma omp barrier
#ifdef profiling
      QValue tnow = time_in_mill_now();
#endif
     // int low = ends[tid];
     // int high = ends[tid + 1];
     // for (int i = low; i < high; ++i) {
    	
     //   indexProcessCRowI(index,
     //       IA[i + 1] - IA[i], JA + IA[i], A + IA[i],
     //       IB, JB, B,
     //       JC + IC[i], C + IC[i]);
     // }
    #pragma omp for schedule(dynamic, stride)
      for(int i=0; i<m; i++) {
          indexProcessCRowI(index,
            IA[i + 1] - IA[i], JA + IA[i], A + IA[i],
            IB, JB, B,
            JC + IC[i], C + IC[i]);
      }
    

#ifdef profiling
     printf("Time passed for thread %d indexProcessCRowI with %lf milliseconds\n", tid, time_in_mill_now() - tnow);
#endif
  }
  free(footPrints);
  //VT profiling
   printf("Time passed for phase 2 somp  is %lf milliseconds\n", time_in_mill_now() - nowl);
   printf("Time passed for total somp  is %lf milliseconds\n", time_in_mill_now() - nowt);

#ifdef profiling
    std::cout << "time passed without memory allocate" << time_in_mill_now() - now << std::endl;
#endif
}

void static_omp_CSR_SpMM(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const int stride) {
#ifdef profiling
    QValue now = time_in_mill_now();
#endif
//VT profiling
    QValue nowt = time_in_mill_now();
    int nthreads = 8;
#pragma omp parallel
#pragma omp master
    nthreads = omp_get_num_threads();
    thread_data_t* thread_datas = allocateThreadDatas(nthreads, n);
    static_omp_CSR_SpMM(IA, JA, A, nnzA,
        IB, JB, B, nnzB,
        IC, JC, C, nnzC,
        m, k, n, thread_datas, stride);
    freeThreadDatas(thread_datas, nthreads);
    
#ifdef profiling
    std::cout << "time passed for static_omp_CSR_SpMM total " <<  time_in_mill_now() - now << std::endl;
#endif
}

void static_omp_CSR_RMCL_OneStep(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const thread_data_t* thread_datas, const int stride) {
  IC = (int*)malloc((m + 1) * sizeof(int));
  int* rowsNnz = (int*)malloc((m + 1) * sizeof(int));
  int* footPrints = (int*)malloc((m + 1) * sizeof(int));
  static int ends[MAX_THREADS_NUM];
  QValue now;
#pragma omp parallel firstprivate(stride)
    {
      const int tid = omp_get_thread_num();
      const int nthreads = omp_get_num_threads();
      dynamic_omp_CSR_IC_nnzC_footprints(IA, JA, IB, JB, m, n, thread_datas[tid], IC, nnzC, footPrints, stride);
#pragma omp barrier
#pragma omp single
      {
#ifdef profiling
        QValue now = time_in_mill_now();
#endif
        arrayEqualPartition(footPrints, m, nthreads, ends);
#ifdef profiling
        std::cout << "time passed for just partition " << time_in_mill_now() - now << std::endl;
        arrayOutput("ends partitions ", stdout, ends, nthreads + 1);
        printf("Footprints partitions\n");
        for (int i = 0; i < nthreads; ++i) {
          printf("%d ", footPrints[ends[i + 1]] - footPrints[ends[i]]);
        }
        printf("\n");
#endif
      }
#pragma omp master
      {
        JC = (int*)malloc(sizeof(int) * nnzC);
        C = (QValue*)malloc(sizeof(QValue) * nnzC);
#ifdef profiling
        now = time_in_mill_now();
#endif
      }
      QValue *x = thread_datas[tid].x;
      int *index = thread_datas[tid].index;
      memset(index, -1, n * sizeof(int));
#pragma omp barrier
#ifdef profiling
      QValue tnow = time_in_mill_now();
#endif
      int low = ends[tid];
      int high = ends[tid + 1];
      for (int i = low; i < high; ++i) {
        QValue *cQValues = C + IC[i];
        int *cColInd = JC + IC[i];
        indexProcessCRowI(index,
            IA[i + 1] - IA[i], JA + IA[i], A + IA[i],
            IB, JB, B,
            JC + IC[i], C + IC[i]);
        int count = IC[i + 1] - IC[i];
        arrayInflationR2(cQValues, count, cQValues);
        pair<QValue, QValue> maxSum = arrayMaxSum(cQValues, count);
        QValue rmax = maxSum.first, rsum = maxSum.second;
        QValue thresh = computeThreshold(rsum / count, rmax);
        arrayThreshPruneNormalize(thresh, cColInd, cQValues,
            &count, cColInd, cQValues);
        rowsNnz[i] = count;
      }
#ifdef profiling
      printf("SOMP time passed for thread %d indexProcessCRowI with %lf milliseconds\n", tid, time_in_mill_now() - tnow);
#endif
#pragma omp barrier
      omp_matrix_relocation(rowsNnz, m, tid, stride, IC, JC, C, nnzC);
    }
    free(footPrints);
    //matrix_relocation(rowsNnz, m, IC, JC, C, nnzC);
    free(rowsNnz);
#ifdef profiling
    std::cout << "static_omp_CSR_RMCL_OneStep(SOMP) time passed " << time_in_mill_now() - now << std::endl;
#endif
}

void static_fair_CSR_RMCL_OneStep(const int IA[], const int JA[], const QValue A[], const int nnzA,
        const int IB[], const int JB[], const QValue B[], const int nnzB,
        int* &IC, int* &JC, QValue* &C, int& nnzC,
        const int m, const int k, const int n, const int stride) {
#ifdef profiling
  QValue now = time_in_mill_now();
#endif
  static_omp_CSR_SpMM(IA, JA, A, nnzA,
      IB, JB, B, nnzB,
      IC, JC, C, nnzC,
      m, k, n, stride);
  int* rowsNnz = (int*)malloc((m + 1) * sizeof(int));
#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
#pragma omp for schedule(dynamic, stride)
    for (int i = 0; i < m; ++i) {
        QValue *cQValues = C + IC[i]; //-1 for one based IC index
        int *cColInd = JC + IC[i];
        int count = IC[i + 1] - IC[i];
        arrayInflationR2(cQValues, count, cQValues);
        pair<QValue, QValue> maxSum = arrayMaxSum(cQValues, count);
        QValue rmax = maxSum.first, rsum = maxSum.second;
        QValue thresh = computeThreshold(rsum / count, rmax);
        arrayThreshPruneNormalize(thresh, cColInd, cQValues,
            &count, cColInd, cQValues);
        rowsNnz[i] = count;
    }
    omp_matrix_relocation(rowsNnz, m, tid, stride, IC, JC, C, nnzC);
  }
  //matrix_relocation(rowsNnz, m, IC, JC, C, nnzC);
  free(rowsNnz);
#ifdef profiling
    std::cout << "static_fair_CSR_RMCL_OneStep(SFOMP) time passed " << time_in_mill_now() - now << std::endl;
#endif
}
