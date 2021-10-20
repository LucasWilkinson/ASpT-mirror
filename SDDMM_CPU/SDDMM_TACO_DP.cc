#include "CSR.h"

#include "mkl.h"
#include "mkls/mkl_csr_kernel.h"
#include "tools/util.h"
#include "tools/ntimer.h"
#include "process_args.h"
#include <time.h>
#include<iomanip>
#include<omp.h>
#define ITERS 1
#define FTYPE double

//-----------COPIED FROM TACO----------------------------------------------
#ifndef TACO_C_HEADERS
#define TACO_C_HEADERS
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include<iostream>
using namespace std;

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define CEIL(a,b) (((a)+(b)-1)/(b))


#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#ifndef TACO_TENSOR_T_DEFINED
#define TACO_TENSOR_T_DEFINED
typedef enum { taco_dim_dense, taco_dim_sparse } taco_dim_t;
typedef struct {
    int     order;      // tensor order (number of dimensions)
    int*    dims;       // tensor dimensions
    taco_dim_t* dim_types;  // dimension storage types
    int     csize;      // component size
    int*    dim_order;  // dimension storage order
    int***  indices;    // tensor index data (per dimension)
    FTYPE*    vals;       // tensor values
} taco_tensor_t;
#endif
#endif


void cacheFlush(FTYPE *X, FTYPE *Y) {
  
  for(int i=0; i<20*1000000; i++) {
    X[i]=Y[i]+rand() % 5;
    Y[i] += rand() % 7;

  }

}


int assemble(taco_tensor_t *O, taco_tensor_t *A, taco_tensor_t *D) {
    int O1_size = *(int*)(O->indices[0][0]);
    int O2_size = *(int*)(O->indices[1][0]);
    FTYPE* __restrict O_vals = (FTYPE*)(O->vals);
    
    O_vals = (FTYPE*)malloc(sizeof(FTYPE) * (O1_size * O2_size));
    
    O->vals = (FTYPE*)O_vals;
    return 0;
}

int compute1(taco_tensor_t *O, taco_tensor_t *A, taco_tensor_t *D) {
    int O1_size = *(int*)(O->indices[0][0]);
    int O2_size = *(int*)(O->indices[1][0]);
    //cout<<"\n11\n";
    FTYPE* __restrict O_vals = (FTYPE*)(O->vals);
    int A1_size = *(int*)(A->indices[0][0]);
    //cout<<"\nA1_size "<<A1_size;
    int* __restrict A2_pos = (int*)(A->indices[1][0]);
    int* __restrict A2_idx = (int*)(A->indices[1][1]);
    FTYPE* __restrict A_vals = (FTYPE*)(A->vals);
    //cout<<"\n13\n";
    int D1_size = *(int*)(D->indices[0][0]);
    int D2_size = *(int*)(D->indices[1][0]);
    //cout<<"\n14\n";
    FTYPE* __restrict D_vals = (FTYPE*)(D->vals);
    //cout<<"\n15\n"<<"O1_size"<<O1_size<<"O2_size"<<O2_size<<endl;
    for (int pO = 0; pO < (O1_size * O2_size); pO++) {
        O_vals[pO] =  (FTYPE)(rand()%1048576)/1048576;
    }
    //cout<<"\nbefore parallel for\n";
     
    return 0;
}


int compute2(taco_tensor_t *O, taco_tensor_t *A, FTYPE *B, taco_tensor_t *D) {
    int O1_size = *(int*)(O->indices[0][0]);
    int O2_size = *(int*)(O->indices[1][0]);
    //cout<<"\n11\n";
    FTYPE* __restrict O_vals = (FTYPE*)(O->vals);
    int A1_size = *(int*)(A->indices[0][0]);
    //cout<<"\nA1_size "<<A1_size;
    int* __restrict A2_pos = (int*)(A->indices[1][0]);
    int* __restrict A2_idx = (int*)(A->indices[1][1]);
    FTYPE* __restrict A_vals = (FTYPE*)(A->vals);
    //cout<<"\n13\n";
    int D1_size = *(int*)(D->indices[0][0]);
    int D2_size = *(int*)(D->indices[1][0]);
    //cout<<"\n14\n";
    FTYPE* __restrict D_vals = (FTYPE*)(D->vals);
    //cout<<"\nbefore parallel for\n";
    #pragma omp parallel for
    for (int mA = 0; mA < A1_size; mA++) {
        
        
        for (int pA2 = A2_pos[mA]; pA2 < A2_pos[mA + 1]; pA2++) {
            int kA = A2_idx[pA2];
            for (int nD = 0; nD < D2_size; nD++) {
                int pD2 = (kA * D2_size) + nD;
                int pO2 = (mA * O2_size) + nD;

                B[pA2] += O_vals[pO2] * (1 * D_vals[pD2]);                
                
            }
		B[pA2] *= A_vals[pA2];
        }
    }
    
    return 0;
}
//-----------------------------------------------------------------------------------------------------------
void initWithDenseMatrixN(const FTYPE * dvalues, const int rows, const int cols, CSR *X) {
    X->rows = rows; X->cols = cols;
    X->rowPtr = (int*)malloc((rows + 1) * sizeof(int));
    X->rowPtr[0] = 0;
    X->nnz = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            FTYPE val = dvalues[i * cols + j];
            if (val < 1e-8) {
                continue;
            }
            X->nnz+=1;
        }
        X->rowPtr[i + 1] = X->nnz;
    }
    X->colInd = (int*)malloc(X->nnz * sizeof(int));
    X->values = (QValue*)malloc(X->nnz * sizeof(QValue));
    int top = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            FTYPE val = dvalues[i * cols + j];
            if (val < 1e-8) {
                continue;
            }
            X->colInd[top] = j;
            X->values[top++] = val;
        }
    }
}

int main(int argc, char *argv[]) {
    process_args(argc, argv);
    print_args();
    COO cooAt;
    cooAt.readSNAPFile(options.inputFileName, false);
    cooAt.orderedAndDuplicatesRemoving();
    CSR A = cooAt.toCSR();
    //cooAt.dispose();
    A.makeOrdered();
    //CSR B = A.deepCopy();
    cout << "\n";
    //omp_set_num_threads(256);
    cout << "  Number of processors available = " << omp_get_num_procs ( ) << "\n";
    cout << "  Number of threads =              " << omp_get_max_threads ( ) << "\n";
    
    int m = A.rows;
    int k = A.cols;
    int n = 0;
    
    cout<<"m "<<m<<" k "<<k<<" n "<<n<<" \n1\n";
    
    struct timeval starttime, midtime, endtime,timediff;
    long double total_time=0.0, avg_time=0.0;
   
    taco_tensor_t *tempA = new taco_tensor_t;//sparse tensor csr
    
    tempA->indices = new int** [2];
    tempA->indices[0] = new int*[2];
    tempA->indices[1] = new int*[2];
    tempA->indices[0][0] = (int*)new int;
    
     *(int*)(tempA->indices[0][0]) = m ;//A1_size num of rows
    
    int* tempA_rowPtr = new int[m+1];
    for(int i = 0; i < m+1 ; i ++)
        tempA_rowPtr[i] = A.rowPtr[i];
    tempA->indices[1][0] = (int*)tempA_rowPtr; //row ptr num of rows + 1 A2_pos
    
    
    int* tempA_colInd = new int[A.nnz];
    for(int i = 0; i < A.nnz ; i ++)
        tempA_colInd[i] = A.colInd[i];
    tempA->indices[1][1] = (int*)tempA_colInd; // col index #nnz A2_idx
    
    
    FTYPE* tempA_vals = new FTYPE[A.nnz];
    for(int i=0;i<A.nnz;i++) {
        tempA_vals[i] = (FTYPE)A.values[i];
    }
    tempA->vals = (FTYPE*)tempA_vals;

     FTYPE* tempB_vals = new FTYPE[A.nnz];
    for(int i=0;i<A.nnz;i++) {
        tempB_vals[i] = 0;
    }
   
   
       FILE *fpo = fopen("SDDMM_KNL_DP.out", "a");
 
    
        cout<<"\n4\n";
    
      //memory allocation for cache-flushing
       //FTYPE *X = (FTYPE*)malloc(20*1000000*sizeof(FTYPE));
       //FTYPE *Y = (FTYPE*)malloc(20*1000000*sizeof(FTYPE));
    int Narray[]={8,32,128};
    for( int ix=1 ; ix <= 2 ; ix++)
    {
        n=Narray[ix];
        //cacheFlush(X, Y);
        
        taco_tensor_t *tempD = new taco_tensor_t;//dense tensor
        tempD->indices = new int** [2];
        tempD->indices[0] = new int*;
        tempD->indices[1] = new int*;
        tempD->indices[0][0] = (int*)new int;
        tempD->indices[1][0] = (int*)new int;
        *(int*)(tempD->indices[0][0]) = k ;
        *(int*)(tempD->indices[1][0]) = n ;
        FTYPE* tempD_vals = new FTYPE[k * n];
        #pragma omp parallel for
        for(int i=0;i<k*n;i++) {
            tempD_vals[i] = (FTYPE)(rand()%1048576)/1048576;
        }
        tempD->vals = (FTYPE*)tempD_vals;
        
        taco_tensor_t *out = new taco_tensor_t; //dense tensor
        out->indices = new int** [2];
        out->indices[0] = new int*;
        out->indices[1] = new int*;
        out->indices[0][0] = (int*)new int;
        out->indices[1][0] = (int*)new int;
        *(int*)(out->indices[0][0]) = m ;
        *(int*)(out->indices[1][0]) = n ;
        FTYPE* out_vals = new FTYPE[m * n];
        #pragma omp parallel for
        for(int i=0;i<m*n;i++) {
            out_vals[i] = (FTYPE)0.0;
        }
        out->vals = (FTYPE*)out_vals;
        
        compute1(out, tempA, tempD);
        gettimeofday(&starttime,NULL);

        compute2(out, tempA, tempB_vals, tempD);
        
        gettimeofday(&endtime,NULL);
        long double elapsed = ((endtime.tv_sec-starttime.tv_sec)*1000000 + endtime.tv_usec-starttime.tv_usec)/1000000.0;
        //printf("n= %d, Elapsed: %llf  seconds\n", n, elapsed);
        //cout<<"taco n= "<<n<<" Elapsed: "<<elapsed<<" sec GFLOPS= "<<(FTYPE)2*(FTYPE)A.nnz*(FTYPE)n/elapsed/1000000000<<endl;
        //cout<<"TTAAGG,"<<options.inputFileName<<","<<n<<","<<(double)2*(double)A.nnz*(double)n/elapsed/1000000000<<endl;
        cout<<"TTAAGG,"<<options.inputFileName<<","<<n<<","<<(double)2*(double)A.nnz*(double)n/elapsed/1000000000<<endl;
        double kkk = (double)2*(double)A.nnz*(double)n/elapsed/1000000000;
        if(n > 8) fprintf(fpo, "%f,", kkk);

        
       

    }
	fclose(fpo);
    return 0;
    
  
}


