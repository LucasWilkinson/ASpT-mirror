#include "CSR.h"

#include "mkl.h"
#include "mkls/mkl_csr_kernel.h"
#include "tools/util.h"
#include "tools/ntimer.h"
#include "process_args.h"
#include <time.h>
#include <stdio.h>
#define ITERS 1
#define FTYPE float
 
void cacheFlush(QValue *X, QValue *Y) {
  
  for(int i=0; i<20*1000000; i++) {
    X[i]=Y[i]+rand() % 5;
    Y[i] += rand() % 7;

  }

}

int main(int argc, char *argv[]) {
	FILE *fpo = fopen("SpMM_KNL_SP.out", "a");

    process_args(argc, argv);
    print_args();
    COO cooAt;
    cooAt.readSNAPFile(options.inputFileName, false);
    cooAt.orderedAndDuplicatesRemoving();
    CSR A = cooAt.toCSR();
    cooAt.dispose();
    //CSR B = A.deepCopy();
    
fprintf(stdout, "TTAAGG,%s,",argv[1]);

//fprintf(stdout, "(%d %d %d)", A.rows, A.cols, A.nnz);

    int m = A.rows;
    int k = A.cols;
    int n = 0;
    

    double mean = 0;
    double sd = 0;
////    cout<<"Average ele in rows : "<<A.nnz/A.rows<<endl;

    for (int mA = 0; mA < m; mA++) {
            
                mean+= (A.rowPtr[mA + 1] - A.rowPtr[mA]);
            
        }
    mean = mean/A.rows;
////    cout<<"\n Mean : "<<mean<<endl;

    double sumOfSquare = 0;
    for (int mA = 0; mA < m; mA++) {
            
                double square_dist = (A.rowPtr[mA + 1] - A.rowPtr[mA]) - mean ;
                if (square_dist < 0)
                    square_dist = square_dist * (-1);

                sumOfSquare += pow(square_dist,2);
            
        }

    sd = sumOfSquare/A.rows;
    sd = sqrt(sd);

////    cout<<" \nstandard daviation = "<< sd<<endl;


    FTYPE         alpha = 1.0, beta = 0.0;
    char          transa, uplo, nonunit;
    char          matdescra[6];
    MKL_INT       i, j, is;
    
    transa = 'N';
    
    //double *B = (double *)malloc(sizeof(double)*k*n);
    //double *C = (double *)malloc(sizeof(double)*m*n);
    
    
    
    
    matdescra[0] = 'G';
    matdescra[1] = 'L';
    matdescra[2] = 'N';
    matdescra[3] = 'C';
    FTYPE *Aval = (FTYPE *)malloc(sizeof(FTYPE)*A.nnz);
    for(int i = 0 ; i< A.nnz ; i++)
        Aval[i] = A.values[i];
    
    struct timeval starttime, midtime, endtime,timediff;
    long double total_time=0.0, avg_time=0.0;
    
    //memory allocation for cache-flushing
   QValue *X = (QValue*)malloc(20*1000000*sizeof(QValue));
   QValue *Y = (QValue*)malloc(20*1000000*sizeof(QValue));
    
    
    
    for( n=8 ; n <= 128 ; n*=4) 
    {
        //cacheFlush(X, Y);
        
        FTYPE *B = (FTYPE *)malloc(sizeof(FTYPE)*k*n);
        FTYPE *C = (FTYPE *)malloc(sizeof(FTYPE)*m*n);
        for(int i=0;i<k*n;i++) {
            B[i] = (FTYPE)(rand()%1048576)/1048576;
        }
        for(int i=0;i<m*n;i++) {
            //C[i] = 0.0;
        }
        gettimeofday(&starttime,NULL);
        
        //mkl_dcsrmm(&transa, &m, &n, &k, &alpha, matdescra, Aval, A.colInd, A.rowPtr,  &(A.rowPtr[1]), &(B[0]), &n,  &beta, &(C[0]), &n);
        mkl_scsrmm(&transa, &m, &n, &k, &alpha, matdescra, Aval, A.colInd, A.rowPtr,  &(A.rowPtr[1]), &(B[0]), &n,  &beta, &(C[0]), &n);
        
        gettimeofday(&endtime,NULL);
        long double elapsed = ((endtime.tv_sec-starttime.tv_sec)*1000000 + endtime.tv_usec-starttime.tv_usec)/1000000.0;
        //long double elapsed2 = ((endtime.tv_sec-midtime.tv_sec)*1000000 + endtime.tv_usec-midtime.tv_usec)/1000000.0;
        //long double elapsed1 = ((midtime.tv_sec-starttime.tv_sec)*1000000 + midtime.tv_usec-starttime.tv_usec)/1000000.0;
        //printf("n= %d , Elapsed: %llf   sec \n", n, elapsed);
//        cout<<"mkl n= "<<n<<" NNZ "<<A.nnz<<" Elapsed: "<<elapsed<<" sec GFLOPS= "<<(double)2*(double)A.nnz*(double)n/elapsed/1000000000<<endl;
	//cout<<"TTAAGG,"<<n<<","<<elapsed*1000<<","<<(double)2*(double)A.nnz*(double)n/elapsed/1000000000<<endl;
	//cout<<"TTAAGG"<<n<<","<<(double)2*(double)A.nnz*(double)n/elapsed/1000000000<<",";


/////        cout<<"TTAAGG,"<<argv[1]<<","<<n<<","<<(double)2*(double)A.nnz*(double)n/elapsed/1000000000<<","<<endl;

	cout<<(double)2*(double)A.nnz*(double)n/elapsed/1000000000<<",";
	double kkk = (double)2*(double)A.nnz*(double)n/elapsed/1000000000;
	if(n > 8) fprintf(fpo, "%f,",kkk);

        // double *C = (double *)malloc(sizeof(double)*m*n);
        // for(int i=0;i<k*n;i++) {
        //     C[i] = 0.0;
        // }
        // matdescra[0] = 'G';
        // matdescra[1] = 'L';
        // matdescra[2] = 'N';
        // matdescra[3] = 'C';
        // mkl_dcsrmm(&transa, &m, &n, &k, &alpha, matdescra, Aval, A.colInd, A.rowPtr,  &(A.rowPtr[1]), &(B[0]), &n,  &beta, &(C[0]), &n);
        
        // gettimeofday(&endtime,NULL);
        // long double elapsed2 = ((endtime.tv_sec-starttime.tv_sec)*1000000 + endtime.tv_usec-starttime.tv_usec)/1000000.0;
        // //long double elapsed2 = ((endtime.tv_sec-midtime.tv_sec)*1000000 + endtime.tv_usec-midtime.tv_usec)/1000000.0;
        // //long double elapsed1 = ((midtime.tv_sec-starttime.tv_sec)*1000000 + midtime.tv_usec-starttime.tv_usec)/1000000.0;
        // //printf("n= %d , Elapsed: %llf   sec \n", n, elapsed);
        // cout<<"mkl n= "<<n<<" NNZ "<<A.nnz<<" Elapsed: "<<elapsed2<<" sec GFLOPS= "<<(double)2*(double)A.nnz*(double)n/elapsed2/1000000000<<endl;


        free (B);
        free (C);
    }
    cout<<endl;
	fclose(fpo);
    return 0;
}


