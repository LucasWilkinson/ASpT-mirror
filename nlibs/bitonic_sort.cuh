#include "tools/macro.h"

__inline__ __device__
void coex(int *keyA,
          int *keyB,
          const int dir)
{
    int t;

    if ((*keyA > *keyB) == dir)
    {
        t = *keyA;
        *keyA = *keyB;
        *keyB = t;
    }
}

__inline__ __device__
void oddeven(int *s_key, int arrayLength)
{
    int dir = 1;

    for (int size = 2; size <= arrayLength; size <<= 1)
    {
        int stride = size >> 1;
        int offset = threadIdx.x & (stride - 1);

        {
            __syncthreads();
            int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            coex(&s_key[pos], &s_key[pos + stride], dir);

            stride >>= 1;
        }

        for (; stride > 0; stride >>= 1)
        {
            __syncthreads();
            int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            if (offset >= stride)
                coex(&s_key[pos - stride], &s_key[pos], dir);
        }
    }
}

__inline__ __device__
void coex(int *keyA,QValue *valA,
          int *keyB,QValue *valB,
          const int dir) 
{
    int t;
    QValue v;

    if ((*keyA > *keyB) == dir) 
    {    
        t = *keyA;
        *keyA = *keyB;
        *keyB = t; 
        v = *valA;
        *valA = *valB;
        *valB = v; 
    }    
}

__inline__ __device__
void oddeven1(int *s_key,QValue *s_val,
             int arrayLength,int threadid)
{
    int dir = 1; 
    for (int size = 2; size <= arrayLength; size <<= 1)
    {    
        int stride = size >> 1;
        int offset = threadid & (stride - 1);
        {    
            __syncthreads();
            int pos = 2 * threadid - (threadid & (stride - 1)); 
            coex(&s_key[pos], &s_val[pos], &s_key[pos + stride], &s_val[pos + stride], dir);
            stride >>= 1;
        }    

        for (; stride > 0; stride >>= 1)
        {    
            __syncthreads();
            int pos = 2 * threadid - (threadid & (stride - 1)); 
            if (offset >= stride)
                coex(&s_key[pos - stride], &s_val[pos - stride], &s_key[pos], &s_val[pos], dir);
        }    
    }    
}


__inline__ __device__
void oddeven1(int *s_key,QValue *s_val,
             int arrayLength,int threadid,int param)
{
    int dir = 1; 
    for (int size = 2; size <= arrayLength; size <<= 1)
    {    
        int stride = size >> 1;
        int offset = threadid & (stride - 1);
        {    
            __syncthreads();
            int pos = 2 * threadid - (threadid & (stride - 1)); 
            coex(&s_key[pos], &s_val[pos], &s_key[pos + stride], &s_val[pos + stride], dir);
            stride >>= 1;
        }    

        for (; stride > 0; stride >>= 1)
        {    
            __syncthreads();
            int pos = 2 * threadid - (threadid & (stride - 1)); 
            if (offset >= stride)
                coex(&s_key[pos - stride], &s_val[pos - stride], &s_key[pos], &s_val[pos], dir);
        }    
    }    
}

__inline__ __device__
void oddeven(int *s_key,QValue *s_val,
             int arrayLength)
{
    int dir = 1; 

    for (int size = 2; size <= arrayLength; size <<= 1)
    {    
        int stride = size >> 1;
        int offset = threadIdx.x & (stride - 1);
        {    
            __syncthreads();
            int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1)); 
            coex(&s_key[pos], &s_val[pos], &s_key[pos + stride], &s_val[pos + stride], dir);
            stride >>= 1;
        }    

        for (; stride > 0; stride >>= 1)
        {    
            __syncthreads();
            int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1)); 
            if (offset >= stride)
                coex(&s_key[pos - stride], &s_val[pos - stride], &s_key[pos], &s_val[pos], dir);
        }    
    }    
}
