#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "cusparse.h"
#define FTYPE double

#define CLEANUP(s)                                   \
do {                                                 \
    printf ("%s\n", s);                              \
    if (yHostPtr)           free(yHostPtr);          \
    if (zHostPtr)           free(zHostPtr);          \
    if (xIndHostPtr)        free(xIndHostPtr);       \
    if (xValHostPtr)        free(xValHostPtr);       \
    if (cooRowIndexHostPtr) free(cooRowIndexHostPtr);\
    if (cooColIndexHostPtr) free(cooColIndexHostPtr);\
    if (cooValHostPtr)      free(cooValHostPtr);     \
    if (y)                  cudaFree(y);             \
    if (z)                  cudaFree(z);             \
    if (xInd)               cudaFree(xInd);          \
    if (xVal)               cudaFree(xVal);          \
    if (csrRowPtr)          cudaFree(csrRowPtr);     \
    if (cooRowIndex)        cudaFree(cooRowIndex);   \
    if (cooColIndex)        cudaFree(cooColIndex);   \
    if (cooVal)             cudaFree(cooVal);        \
    if (handle)             cusparseDestroy(handle); \
    fflush (stdout);                                 \
} while (0)

struct v_struct {
        int row, col;
        FTYPE val;
};

int *csr_v, *csr_e;
FTYPE *csr_ev;

double rtclock(void)
{
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d",stat);
  return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

/*
int compare1(const void *a, const void *b)
{
        if (((struct v_struct *)a)->src - ((struct v_struct *)b)->src > 0) return 1;
        if (((struct v_struct *)a)->src - ((struct v_struct *)b)->src < 0) return -1;
        return ((struct v_struct *)a)->dst - ((struct v_struct *)b)->dst;
}*/

int compare1(const void *a, const void *b)
{
        if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row > 0) return 1;
        if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row < 0) return -1;
        return ((struct v_struct *)a)->col - ((struct v_struct *)b)->col;
}


int main(int argc, char **argv)
{
	FILE *fpo = fopen("SpMM_GPU_DP.out", "a");
	FILE *fp;
	srand(time(NULL));
//fprintf(stdout,"TTAAGG,%s,",argv[1]);
    cudaError_t cudaStat1,cudaStat2,cudaStat3,cudaStat4,cudaStat5,cudaStat6;
    cusparseStatus_t status;
    cusparseHandle_t handle=0;
    cusparseMatDescr_t descra=0;
/*    int *    cooRowIndexHostPtr=0;
    int *    cooColIndexHostPtr=0;    
    double * cooValHostPtr=0;
    int *    cooRowIndex=0;
    int *    cooColIndex=0;    
    double * cooVal=0;
    int *    xIndHostPtr=0;
    double * xValHostPtr=0;
    double * yHostPtr=0;
    int *    xInd=0;
    double * xVal=0;
    double * y=0;  
    int *    csrRowPtr=0;
    double * zHostPtr=0; 
    double * z=0; */
    int      n, nc, nnz, nflag, nnz_vector, i, j;

	int nr, sc;

	struct v_struct *temp_v;

//    printf("testing example\n");
    /* create the following sparse test matrix in COO format */
    /* |1.0     2.0 3.0|
       |    4.0        |
       |5.0     6.0 7.0|
       |    8.0     9.0| */
//    n=4; nnz=9; 

	char buf[300];
	int sflag;
	int dummy, pre_count=0, tmp_ne;

	fp = fopen(argv[1], "r");
	//sc = atoi(argv[2]);
	fgets(buf, 300, fp);

        if(strstr(buf, "symmetric") != NULL || strstr(buf, "Hermitian") != NULL) sflag = 1; // symmetric
        else sflag = 0;
        if(strstr(buf, "pattern") != NULL) nflag = 0; // non-value
        else if(strstr(buf, "complex") != NULL) nflag = -1;
        else nflag = 1;

        while(1) {
                pre_count++;
                fgets(buf, 300, fp);
                if(strstr(buf, "%") == NULL) break;
        }
        fclose(fp);

        fp = fopen(argv[1], "r");
        for(i=0;i<pre_count;i++)
                fgets(buf, 300, fp);

        fscanf(fp, "%d %d %d", &nr, &nc, &nnz);
        nnz *= (sflag+1);

//fprintf(stdout, "config : %d %d %d\n", nr, nc, nnz);

        temp_v = (struct v_struct *)malloc(sizeof(struct v_struct)*(nnz+1));

        for(i=0;i<nnz;i++) {
                fscanf(fp, "%d %d", &temp_v[i].row, &temp_v[i].col);
                temp_v[i].row--; temp_v[i].col--;

                if(temp_v[i].row < 0 || temp_v[i].row >= nr || temp_v[i].col < 0 || temp_v[i].col >= nc) {
                        fprintf(stdout, "A vertex id is out of range %d %d\n", temp_v[i].row, temp_v[i].col);
                        exit(0);
                }
                if(nflag == 0) temp_v[i].val = (FTYPE)(rand()%1048576)/1048576;
                else if(nflag == 1) {
                        FTYPE ftemp;
                        fscanf(fp, " %f ", &ftemp);
                        temp_v[i].val = ftemp;
                } else { // complex
                        FTYPE ftemp1, ftemp2;
                        fscanf(fp, " %f %f ", &ftemp1, &ftemp2);
                        temp_v[i].val = ftemp1;
                }
                if(sflag == 1) {
                        i++;
                        temp_v[i].row = temp_v[i-1].col;
                        temp_v[i].col = temp_v[i-1].row;
                        temp_v[i].val = temp_v[i-1].val;
                }
        }
        qsort(temp_v, nnz, sizeof(struct v_struct), compare1);

	int *loc;
        loc = (int *)malloc(sizeof(int)*(nnz+1));

        memset(loc, 0, sizeof(int)*(nnz+1));
        loc[0]=1;
        for(i=1;i<nnz;i++) {
                if(temp_v[i].row == temp_v[i-1].row && temp_v[i].col == temp_v[i-1].col)
                        loc[i] = 0;
                else loc[i] = 1;
        }
        for(i=1;i<=nnz;i++)
                loc[i] += loc[i-1];
        for(i=nnz; i>=1; i--)
                loc[i] = loc[i-1];
        loc[0] = 0;

        for(i=0;i<nnz;i++) {
                temp_v[loc[i]].row = temp_v[i].row;
                temp_v[loc[i]].col = temp_v[i].col;
                temp_v[loc[i]].val = temp_v[i].val;
        }
        nnz = loc[nnz];

//fprintf(stderr, "nr : %d nc : %d nnz : %d\n", nr, nc, nnz);


//	fscanf(fp, "%d %d %d", &n, &nnz, &nflag);
//	temp_v = (struct v_struct *)malloc(sizeof(struct v_struct)*nnz);
//	for(i=0;i<nnz;i++) {
//		fscanf(fp, "%d %d", &temp_v[i].src, &temp_v[i].dst);
//		if(nflag == 1) fscanf(fp, "%f", &temp_v[i].val);
//		else temp_v[i].val = ((FTYPE)(rand()%1048576))/1048576;
//	}
        csr_v = (int *)malloc(sizeof(int)*(nr+1));
        csr_e = (int *)malloc(sizeof(int)*nnz);
        csr_ev = (FTYPE *)malloc(sizeof(FTYPE)*nnz);
//	qsort(temp_v, nnz, sizeof(struct v_struct), compare1);
 	for(i=0;i<=nr;i++)
		csr_v[i]=0;
	int temp_point=0;
	csr_v[0]=0;
	csr_v[nr] = nnz;
	for(i=0;i<nnz;i++) {
		csr_e[i] = temp_v[i].col;
		csr_ev[i] = temp_v[i].val;
		csr_v[1+temp_v[i].row] = i+1;
		//if(csr_v[temp_v[i].row] == -1)
		//	csr_v[temp_v[i].row] = i;
	}
	for(i=1;i<nr; i++)
		if(csr_v[i] == 0) csr_v[i] = csr_v[i-1];
	int *ccsr_v, *ccsr_e; FTYPE *ccsr_ev;

	cudaMalloc((void **) &ccsr_v, sizeof(int)*(nr+1));
	cudaMalloc((void **) &ccsr_e, sizeof(int)*nnz);
	cudaMalloc((void **) &ccsr_ev, sizeof(FTYPE)*nnz);
	cudaMemcpy(ccsr_v, csr_v, sizeof(int)*(nr+1), cudaMemcpyHostToDevice);
	cudaMemcpy(ccsr_e, csr_e, sizeof(int)*(nnz), cudaMemcpyHostToDevice);
	cudaMemcpy(ccsr_ev, csr_ev, sizeof(FTYPE)*(nnz), cudaMemcpyHostToDevice);
	
	

//////////////
/*
    cooRowIndexHostPtr = (int *)   malloc(nnz*sizeof(cooRowIndexHostPtr[0])); 
    cooColIndexHostPtr = (int *)   malloc(nnz*sizeof(cooColIndexHostPtr[0])); 
    cooValHostPtr      = (double *)malloc(nnz*sizeof(cooValHostPtr[0])); 
    if ((!cooRowIndexHostPtr) || (!cooColIndexHostPtr) || (!cooValHostPtr)){
        CLEANUP("Host malloc failed (matrix)");
        return EXIT_FAILURE; 
    }
    cooRowIndexHostPtr[0]=0; cooColIndexHostPtr[0]=0; cooValHostPtr[0]=1.0;  
    cooRowIndexHostPtr[1]=0; cooColIndexHostPtr[1]=2; cooValHostPtr[1]=2.0;  
    cooRowIndexHostPtr[2]=0; cooColIndexHostPtr[2]=3; cooValHostPtr[2]=3.0;  
    cooRowIndexHostPtr[3]=1; cooColIndexHostPtr[3]=1; cooValHostPtr[3]=4.0;  
    cooRowIndexHostPtr[4]=2; cooColIndexHostPtr[4]=0; cooValHostPtr[4]=5.0;  
    cooRowIndexHostPtr[5]=2; cooColIndexHostPtr[5]=2; cooValHostPtr[5]=6.0;
    cooRowIndexHostPtr[6]=2; cooColIndexHostPtr[6]=3; cooValHostPtr[6]=7.0;  
    cooRowIndexHostPtr[7]=3; cooColIndexHostPtr[7]=1; cooValHostPtr[7]=8.0;  
    cooRowIndexHostPtr[8]=3; cooColIndexHostPtr[8]=3; cooValHostPtr[8]=9.0;  
    //print the matrix
    printf("Input data:\n");
    for (i=0; i<nnz; i++){        
        printf("cooRowIndexHostPtr[%d]=%d  ",i,cooRowIndexHostPtr[i]);
        printf("cooColIndexHostPtr[%d]=%d  ",i,cooColIndexHostPtr[i]);
        printf("cooValHostPtr[%d]=%f     \n",i,cooValHostPtr[i]);
    }
*/  
    /* create a sparse and dense vector */ 
    /* xVal= [100.0 200.0 400.0]   (sparse)
       xInd= [0     1     3    ]
       y   = [10.0 20.0 30.0 40.0 | 50.0 60.0 70.0 80.0] (dense) */
/*
    nnz_vector = 3;
    xIndHostPtr = (int *)   malloc(nnz_vector*sizeof(xIndHostPtr[0])); 
    xValHostPtr = (double *)malloc(nnz_vector*sizeof(xValHostPtr[0])); 
    yHostPtr    = (double *)malloc(2*n       *sizeof(yHostPtr[0]));
    zHostPtr    = (double *)malloc(2*(n+1)   *sizeof(zHostPtr[0]));
    if((!xIndHostPtr) || (!xValHostPtr) || (!yHostPtr) || (!zHostPtr)){
        CLEANUP("Host malloc failed (vectors)");
        return EXIT_FAILURE; 
    }
    yHostPtr[0] = 10.0;  xIndHostPtr[0]=0; xValHostPtr[0]=100.0; 
    yHostPtr[1] = 20.0;  xIndHostPtr[1]=1; xValHostPtr[1]=200.0;  
    yHostPtr[2] = 30.0;
    yHostPtr[3] = 40.0;  xIndHostPtr[2]=3; xValHostPtr[2]=400.0;  
    yHostPtr[4] = 50.0;
    yHostPtr[5] = 60.0;
    yHostPtr[6] = 70.0;
    yHostPtr[7] = 80.0;
//    yHostPtr[8] = 100.0;
//    yHostPtr[9] = 200.0;
//    yHostPtr[10] = 300.0;
//    yHostPtr[11] = 400.0;
//    yHostPtr[12] = 500.0;
//    yHostPtr[13] = 600.0;
//    yHostPtr[14] = 700.0;
//    yHostPtr[15] = 800.0;

    //print the vectors
    for (j=0; j<2; j++){
        for (i=0; i<n; i++){        
            printf("yHostPtr[%d,%d]=%f\n",i,j,yHostPtr[i+n*j]);
        }
    }
    for (i=0; i<nnz_vector; i++){        
        printf("xIndHostPtr[%d]=%d  ",i,xIndHostPtr[i]);
	printf("xValHostPtr[%d]=%f\n",i,xValHostPtr[i]);
    }
*/
    /* allocate GPU memory and copy the matrix and vectors into it */

/*
    cudaStat1 = cudaMalloc((void**)&cooRowIndex,nnz*sizeof(cooRowIndex[0])); 
    cudaStat2 = cudaMalloc((void**)&cooColIndex,nnz*sizeof(cooColIndex[0]));
    cudaStat3 = cudaMalloc((void**)&cooVal,     nnz*sizeof(cooVal[0])); 
    cudaStat4 = cudaMalloc((void**)&y,          2*n*sizeof(y[0]));   
    cudaStat5 = cudaMalloc((void**)&xInd,nnz_vector*sizeof(xInd[0])); 
    cudaStat6 = cudaMalloc((void**)&xVal,nnz_vector*sizeof(xVal[0])); 
    if ((cudaStat1 != cudaSuccess) ||
        (cudaStat2 != cudaSuccess) ||
        (cudaStat3 != cudaSuccess) ||
        (cudaStat4 != cudaSuccess) ||
        (cudaStat5 != cudaSuccess) ||
        (cudaStat6 != cudaSuccess)) {
        CLEANUP("Device malloc failed");
        return EXIT_FAILURE; 
    }    
    cudaStat1 = cudaMemcpy(cooRowIndex, cooRowIndexHostPtr, 
                           (size_t)(nnz*sizeof(cooRowIndex[0])), 
                           cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(cooColIndex, cooColIndexHostPtr, 
                           (size_t)(nnz*sizeof(cooColIndex[0])), 
                           cudaMemcpyHostToDevice);
    cudaStat3 = cudaMemcpy(cooVal,      cooValHostPtr,      
                           (size_t)(nnz*sizeof(cooVal[0])),      
                           cudaMemcpyHostToDevice);
    cudaStat4 = cudaMemcpy(y,           yHostPtr,           
                           (size_t)(2*n*sizeof(y[0])),           
                           cudaMemcpyHostToDevice);
    cudaStat5 = cudaMemcpy(xInd,        xIndHostPtr,        
                           (size_t)(nnz_vector*sizeof(xInd[0])), 
                           cudaMemcpyHostToDevice);
    cudaStat6 = cudaMemcpy(xVal,        xValHostPtr,        
                           (size_t)(nnz_vector*sizeof(xVal[0])), 
                           cudaMemcpyHostToDevice);
    if ((cudaStat1 != cudaSuccess) ||
        (cudaStat2 != cudaSuccess) ||
        (cudaStat3 != cudaSuccess) ||
        (cudaStat4 != cudaSuccess) ||
        (cudaStat5 != cudaSuccess) ||
        (cudaStat6 != cudaSuccess)) {
        CLEANUP("Memcpy from Host to Device failed");
        return EXIT_FAILURE;
    }
   */ 
    /* initialize cusparse library */
    status= cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        return EXIT_FAILURE;
    }
    /* create and setup matrix descriptor */ 
    status= cusparseCreateMatDescr(&descra); 
    if (status != CUSPARSE_STATUS_SUCCESS) {
        return EXIT_FAILURE;
    }       
    cusparseSetMatType(descra,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descra,CUSPARSE_INDEX_BASE_ZERO);  
    
    /* exercise conversion routines (convert matrix from COO 2 CSR format) */
/*    cudaStat1 = cudaMalloc((void**)&csrRowPtr,(n+1)*sizeof(csrRowPtr[0]));
    if (cudaStat1 != cudaSuccess) {
        CLEANUP("Device malloc failed (csrRowPtr)");
        return EXIT_FAILURE;
    }
    status= cusparseXcoo2csr(handle,cooRowIndex,nnz,n,
                             csrRowPtr,CUSPARSE_INDEX_BASE_ZERO); 
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Conversion from COO to CSR format failed");
	return EXIT_FAILURE;
    }  
*/
    //csrRowPtr = [0 3 4 7 9] 
    /* exercise Level 1 routines (scatter vector elements) */
/*    status= cusparseDsctr(handle, nnz_vector, xVal, xInd, 
                          &y[n], CUSPARSE_INDEX_BASE_ZERO);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Scatter from sparse to dense vector failed");
        return EXIT_FAILURE;
    } */ 
    //y = [10 20 30 40 | 100 200 70 400]
    /* exercise Level 2 routines (csrmv) */
//cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, 2.0, descra, cooVal, csrRowPtr, cooColIndex, &y[0], 3.0, &y[n]);
// status = cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, descra, cooVal, csrRowPtr, cooColIndex, &y[0], 0.0, &y[n]);

    /* exercise Level 3 routines (csrmm) */

	//const double *zz;

   for(sc=32; sc<=128; sc*=4) {
	cudaError_t err = cudaSuccess;
	FTYPE *y_in, *cy_in, *y_out, *cy_out; 
	y_in = (FTYPE *)malloc(sizeof(FTYPE)*nc*sc);
	y_out = (FTYPE *)malloc(sizeof(FTYPE)*(nr+1)*sc);
	for(i=0;i<nc*sc;i++)
		y_in[i] = ((FTYPE)(rand()%1048576))/1048576;
	err = cudaMalloc((void **) &cy_in, sizeof(FTYPE)*nc*sc);
        if(err != cudaSuccess)  {fprintf(stdout, "\n"); exit(0); }
	err = cudaMalloc((void **) &cy_out, sizeof(FTYPE)*(nr+1)*sc);
        if(err != cudaSuccess)  {fprintf(stdout, "\n"); exit(0); }
	cudaMemcpy(cy_in, y_in, sizeof(FTYPE)*nc*sc, cudaMemcpyHostToDevice);
    	cudaMemset((void *)cy_out, 0, sc*(nr+1)*sizeof(FTYPE));    

        float tot_ms;
        cudaEvent_t event1, event2;
        cudaEventCreate(&event1);
        cudaEventCreate(&event2);



	const FTYPE alpha=1.0f, beta=0.0f;

        cudaDeviceSynchronize();
        cudaEventRecord(event1,0);


#define ITER (1)
for(int ik=0;ik<ITER;ik++) {
     cusparseDcsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,  nr, sc, nc, nnz,
                           &alpha, descra, ccsr_ev, ccsr_v, ccsr_e, cy_in, sc,
//     cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, nr, sc, nc, nnz,
//                           &alpha, descra, ccsr_ev, ccsr_v, ccsr_e, cy_in, nr, 
                           &beta,
			 cy_out,
			 nr+1);
}


        cudaEventRecord(event2,0);
        cudaEventSynchronize(event1);
        cudaEventSynchronize(event2);
        cudaEventElapsedTime(&tot_ms, event1, event2);
        cudaDeviceSynchronize();


	cudaMemcpy(y_out, cy_out, sizeof(FTYPE)*(nr+1)*sc, cudaMemcpyDeviceToHost);

//#define VALID
#ifdef VALID
	FTYPE *gold_out = (FTYPE *)malloc(sizeof(FTYPE)*(nr+1)*sc);
	memset(gold_out, 0, sizeof(FTYPE)*nr*sc);
	for(i=0;i<nnz;i++) {
		for(j=0;j<sc;j++) {
			//gold_out[temp_v[i].row + (nr+1)*j] += y_in[temp_v[i].col + nc*j] * temp_v[i].val;
			gold_out[temp_v[i].row + (nr+1)*j] += y_in[sc*temp_v[i].col + j] * temp_v[i].val;
		}
	}

	long num_diff=0;
	for(i=0;i<(nr+1)*sc; i++) {
	//	if(abs(y_out[i] -gold_out[i])/max(abs(y_out[i]), abs(gold_out[i])) > 0.05) printf("diff : %f %f (%d %d)\n", y_out[i], gold_out[i], i, nr*sc);
		if(abs(y_out[i] -gold_out[i])/max(abs(y_out[i]), abs(gold_out[i])) > 0.05) {
			num_diff++;
			if(num_diff < 30) {
		//		fprintf(stdout, "diff : %d\t%f\t%f\n", y_out[i], gold_out[i]);
			}
		}
	}
	fprintf(stdout, "(%ld),",num_diff);	
	//fprintf(stdout, "time(ms) : %f\tGFlops : %f\t%ld\n", (end-start)*1000, (double)nnz*2*sc/(end-start)/1000000000, num_diff);
#endif
	cudaFree(cy_out); cudaFree(cy_in); free(y_out); free(y_in);
        fprintf(fpo, "%f,", (double)ITER*(double)nnz*2*sc/tot_ms/1000000);
	//fprintf(stdout, "%f,", (double)ITER*(double)nnz*2*sc/tot_ms/1000000);
    }
	fclose(fpo);
//    fprintf(stdout, "\n");
//	fprintf(stdout, "%s,%d,%f,%f,%f\n",argv[1] ,sc,(end-start)*1000, (double)nnz*2*sc/(end-start)/1000000000, (double)num_diff/(sc*nr));



/*
    cudaStat1 = cudaMemcpy(zHostPtr, z, 
                           (size_t)(2*(n)*sizeof(z[0])), 
                           cudaMemcpyDeviceToHost);
    if (cudaStat1 != cudaSuccess)  {
        CLEANUP("Memcpy from Device to Host failed");
        return EXIT_FAILURE;
    } 
    //z = [950 400 2550 2600 0 | 49300 15200 132300 131200 0]
    printf("Final results:\n");
    for (j=0; j<2; j++){
        for (i=0; i<n; i++){
            printf("z[%d,%d]=%f\n",i,j,zHostPtr[i+(n)*j]);
        }
    }
*/
    
    /* check the results */
    /* Notice that CLEANUP() contains a call to cusparseDestroy(handle) */
/*
    if ((zHostPtr[0] != 950.0)    || 
        (zHostPtr[1] != 400.0)    || 
        (zHostPtr[2] != 2550.0)   || 
        (zHostPtr[3] != 2600.0)   || 
        (zHostPtr[4] != 0.0)      || 
	(zHostPtr[5] != 49300.0)  || 
        (zHostPtr[6] != 15200.0)  || 
        (zHostPtr[7] != 132300.0) || 
        (zHostPtr[8] != 131200.0) || 
        (zHostPtr[9] != 0.0)      ||
        (yHostPtr[0] != 10.0)     || 
        (yHostPtr[1] != 20.0)     || 
        (yHostPtr[2] != 30.0)     || 
        (yHostPtr[3] != 40.0)     || 
        (yHostPtr[4] != 680.0)    || 
        (yHostPtr[5] != 760.0)    || 
        (yHostPtr[6] != 1230.0)   || 
        (yHostPtr[7] != 2240.0)){ 
//printf("err");
        CLEANUP("example test FAILED");
        return EXIT_FAILURE;
    }
    else{
        CLEANUP("example test PASSED");
        return EXIT_SUCCESS;
    } */     
}


