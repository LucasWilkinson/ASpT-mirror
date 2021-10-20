#ifndef _FRIENDS_H_
#define _FRIENDS_H_

#include <iostream>
#include <algorithm>
#include "bicsb.h"
#include "bmcsb.h"
#include "bmsym.h"
#include "csbsym.h"
#include "utility.h"
#include "timer.gettimeofday.c"

using namespace std;	

template <class NU, class IU>	
class BiCsb;

template <class NU, class IU, unsigned UUDIM>
class BmCsb;

double prescantime;


#if (__GNUC__ == 4 && (__GNUC_MINOR__ < 7) )
#define emplace_back push_back
#endif



// SpMV with Bit-Masked CSB
// No semiring or type promotion support yet
template <typename NT, typename IT, unsigned TTDIM>
void bmcsb_gespmv (const BmCsb<NT, IT, TTDIM> & A, const NT * __restrict x, NT * __restrict y)
{
	double t0 = timer_seconds_since_init();
	
	unsigned * scansum = new unsigned[A.nrb];
	unsigned sum = prescan(scansum, A.masks, A.nrb);
	
	double t1 = timer_seconds_since_init();
	prescantime += (t1-t0);

	IT ysize = A.lowrowmask + 1;			// size of the output subarray (per block row - except the last)

	if( A.isPar() )
	{	
        float rowave = static_cast<float>(A.numnonzeros()) / (A.nbr-1);
		cilk_for (IT i = 0 ; i < A.nbr ; ++i)	// for all block rows of A 
		{
			IT *  btop = A.top [i];		// get the pointer to this block row
			IT rhi = ((i << A.rowlowbits) & A.highrowmask);
			NT * suby = &y[rhi];
			if( A.top[i][A.nbc] - A.top[i][0] >   BALANCETH * rowave)
			{
				IT thsh = ysize * BREAKNRB;   
				vector<IT*> chunks;
				chunks.push_back(btop);
				for(IT j =0; j < A.nbc; )
				{
					IT count = btop[j+1] - btop[j];
					if(count < thsh && j < A.nbc)
					{
						while(count < thsh && j < A.nbc)
						{
							count += btop[(++j)+1] - btop[j]; 
						}
						chunks.push_back(btop+j);	// push, but exclude the block that caused the overflow
					}
					else
					{
						chunks.push_back(btop+(++j));	// don't exclude the overflow block if it is the only block in that chunk
					}
				}
				// In std:vector, the elements are stored contiguously so that we can 
				// treat &chunks[0] as an array of pointers to IT w/out literally copying it to IT**
				if(i==(A.nbr-1))	// last iteration
				{
					A.BMult(&chunks[0], 0, chunks.size()-1, x, suby,  A.rowsize() - ysize*i, scansum);
				}
				else
				{
					A.BMult(&chunks[0], 0, chunks.size()-1, x, suby, ysize, scansum);
				}
			}
			else
			{
				A.SubSpMV(btop, 0, A.nbc, x, suby, scansum);
			}
		}
	}

	else
	{
		cilk_for (IT i = 0 ; i < A.nbr ; ++i)    // for all block rows of A 
                {
			IT * btop = A.top [i];                       // get the pointer to this block row
                        IT rhi = ((i << A.rowlowbits) & A.highrowmask);
                        NT * suby = &y[rhi];

			A.SubSpMV(btop, 0, A.nbc, x, suby, scansum);
		}
	}
	delete [] scansum;
}

/**
  * Operation y = A*x+y on a semiring SR
  * A: a general CSB matrix (no specialization on booleans is necessary as this loop is independent of numerical values) 
  * x: a column vector or a set of column vectors (i.e. array of structs, array of std:arrays, etc))
  * SR::multiply() handles the multiple rhs and type promotions, etc. 
 **/
template <typename SR, typename NT, typename IT, typename RHS, typename LHS>
void bicsb_gespmv (const BiCsb<NT, IT> & A, const RHS * __restrict x, LHS * __restrict y)
{
	IT ysize = A.lowrowmask + 1;			// size of the output subarray (per block row - except the last)

	if(A.isPar() )
	{	
        	float rowave = static_cast<float>(A.numnonzeros()) / (A.nbr-1);
		cilk_for (IT i = 0 ; i < A.nbr ; ++i)	// for all block rows of A 
		{
			IT *  btop = A.top [i];		// get the pointer to this block row
			IT rhi = ((i << A.rowlowbits) & A.highrowmask);
			LHS * suby = &y[rhi];

			if(A.top[i][A.nbc] - A.top[i][0] >  std::max( static_cast<NT>(BALANCETH * rowave), static_cast<NT>(BREAKEVEN * ysize) ) )
			{
				IT thsh = BREAKEVEN * ysize;
				vector<IT*> chunks;
				chunks.push_back(btop);
				for(IT j =0; j < A.nbc; )
				{
					IT count = btop[j+1] - btop[j];
					if(count < thsh && j < A.nbc)
					{
						while(count < thsh && j < A.nbc)
						{
							count += btop[(++j)+1] - btop[j]; 
						}
						chunks.push_back(btop+j);	// push, but exclude the block that caused the overflow
					}
					else
					{
						chunks.push_back(btop+(++j));	// don't exclude the overflow block if it is the only block in that chunk
					}
				}
				// In std:vector, the elements are stored contiguously so that we can 
				// treat &chunks[0] as an array of pointers to IT w/out literally copying it to IT**
				if(i==(A.nbr-1))	// last iteration
				{
					A.template BMult<SR>(&chunks[0], 0, chunks.size()-1, x, suby,  A.rowsize() - ysize*i);
				}
				else
				{
					A.template BMult<SR>(&chunks[0], 0, chunks.size()-1, x, suby, ysize);	// chunksize-1 because we always insert a dummy chunk
				}
			}
			else
			{
				A.template SubSpMV<SR>(btop, 0, A.nbc, x, suby);
			}
		}
	}
	else
	{
		cilk_for (IT i = 0 ; i < A.nbr ; ++i)    // for all block rows of A 
        	{
			IT * btop = A.top [i];                       // get the pointer to this block row
           	 	IT rhi = ((i << A.rowlowbits) & A.highrowmask);
            		LHS * suby = &y[rhi];
			A.template SubSpMV<SR>(btop, 0, A.nbc, x, suby);
		}
	}
}


/**
  * Operation y = (A^t)*x+y a semiring SR
  * A: a general CSB matrix (no specialization on booleans is necessary as this loop is independent of numerical values) 
  * x: a column vector or a set of column vectors (i.e. array of structs, array of std:arrays, etc))
  * SR::multiply() handles the multiple rhs and type promotions, etc. 
  */
template <typename SR, typename NT, typename IT, typename RHS, typename LHS>
void bicsb_gespmvt (const BiCsb<NT, IT> & A, const RHS * __restrict x, LHS * __restrict y)
{
    IT ysize = A.lowcolmask + 1;			// size of the output subarray (per block column - except the last)
    
    // A.top (nbr=3, nbc=4):
    //  0  5 17 21 24
    // 24 28 33 39 53
    // 53 60 61 70 72

    vector<IT> colsums(A.nbc,0);
    cilk_for(IT j=0; j<A.nbc; ++j)
    {
        for(IT i=0; i< A.nbr; ++i)
        {
            colsums[j] += (A.top[i][j+1] - A.top[i][j]);
        }
    }    

    if( A.isPar() )
    {
        float colave = static_cast<float>(A.numnonzeros()) / (A.nbc-1);
        cilk_for (IT j = 0 ; j < A.nbc ; ++j)	// for all block columns of A
        {
            IT rhi = ((j << A.rowlowbits) & A.highcolmask);
            LHS * suby = &y[rhi];
            typedef typename std::tuple<IT,IT,IT> IntTriple;
            typedef typename std::vector< IntTriple > ChunkType;
            vector< ChunkType * >  chunks;    // we will have to manage
            
	    // the second condition is == natural == because if colsums[j] < BREAKEVEN * ysize, 
	    // then the whole row will be a single chunk of sparse blocks that runs as a single strand
            if( colsums[j] >   BALANCETH * colave && colsums[j] > BREAKEVEN * ysize)
            {
                IT thsh = BREAKEVEN * ysize;
                // each chunk is represented by a vector of blocks
                // each block is represented by its {begin, end} pointers to bot array AND its -row- block id (within the block column)
                // get<0>(tuple): begin pointer to bot, get<1>(tuple): end pointer to bot, get<2>(tuple): row block id
            
                for(IT i =0; i < A.nbr; ++i )
                {
		    ChunkType * chunk = new ChunkType();
		    chunk->emplace_back( IntTriple (A.top[i][j], A.top[i][j+1], i));
                    IT count = A.top[i][j+1] - A.top[i][j];
					
                    if(count < thsh)	
                    {
			// while adding the next (i+1) element wouldn't exceed the chunk limit
                        while(i < A.nbr-1 && (count+A.top[i+1][j+1] - A.top[i+1][j]) < thsh )
                        {
                            	i++;    // move to next one before push 
			    	if(A.top[i][j+1] - A.top[i][j] > 0)
			    	{
                            		chunk->emplace_back( IntTriple (A.top[i][j], A.top[i][j+1], i));
                            		count += A.top[i][j+1] - A.top[i][j];
				}
                        }
						// push, but exclude the block that caused the overflow
                        chunks.push_back(chunk);    // emplace_back wouldn't buy anything for simple structures like pointers 
                    }
                    else // already above the limit by itself => single dense block
                    {
                        chunks.push_back(chunk);
                    }
                }
                if(j==(A.nbc-1))	// last iteration
                {
                    A.template BTransMult<SR>(chunks, 0, chunks.size(), x, suby,  A.colsize() - ysize*j);
                }
                else
                {
                    A.template BTransMult<SR>(chunks, 0, chunks.size(), x, suby, ysize); // chunksize (no -1) as there is no dummy chunk
                }
            
                // call the destructor of each chunk vector
                for_each(chunks.begin(), chunks.end(), [](ChunkType * pPtr){ delete pPtr; });
            }
            else
            {
                A.template SubSpMVTrans<SR>(j, 0, A.nbr, x, suby);
            }
        }
    }
    else
    {
        cilk_for (IT j =0; j< A.nbc; ++j)  // for all block columns of A
        {
            IT rhi = ((j << A.collowbits) & A.highcolmask);
   	       	LHS * suby = &y[rhi];

            A.template SubSpMVTrans<SR>(j, 0, A.nbr, x, suby);
        }
    }
}

// SpMV with symmetric CSB
// No semiring or type promotion support yet
template <typename NT, typename IT>
void csbsym_gespmv (const CsbSym<NT, IT> & A, const NT * __restrict x, NT * __restrict y)
{
	#pragma isat marker SM2_begin
	//if(  A.isPar() )
	//{	
		#pragma isat tuning name(tune_tempy) scope(SM1_begin, SM1_end) measure(SM2_begin, SM2_end) variable(SPAWNS, range(1,6)) variable(NDIAGS, range(1,11)) search(dependent)
		#pragma isat marker SM1_begin
		#define SPAWNS 1 	// how many you do in parallel at a time
		#define NDIAGS 3	// how many you do in total
		NT ** t_y = new NT* [SPAWNS];
		t_y[0] = y;	// alias t_y[0] to y
		for(int i=1; i<SPAWNS; ++i)
		{
			t_y[i] = new NT[A.n]();
		}
		if(NDIAGS < SPAWNS)
		{
			cout << "Impossible to execute" << endl;
			return;
		}
		int syncs = NDIAGS / SPAWNS;
		int remdiags = NDIAGS;
		for(int j=0; j < syncs; ++j)
		{
			if(remdiags > 1)
			{
				A.MultDiag(t_y[0], x, j*SPAWNS);	// maps to A.MultMainDiag(y,x) if j = 0
				--remdiags;	// decrease remaining diagonals
				int i = 1;
				for(; (i < SPAWNS) && (remdiags > 1) ; ++i)
				{
					cilk_spawn A.MultDiag(t_y[i], x, j*SPAWNS + i);
					--remdiags;
				}
				if(i < SPAWNS && remdiags == 1)
				{
					cilk_spawn A.MultAddAtomics(t_y[i], x, j*SPAWNS + i);		
					--remdiags;
				}
				cilk_sync;
			}
			else if(remdiags == 1)
			{
				A.MultAddAtomics(t_y[0], x, j*SPAWNS);	// will only happen is remdiags is 1 when the outerloop started
				--remdiags;
			}
		}

		cilk_for(int j=0; j< A.n; ++j)
		{
			for(int i=1; i<SPAWNS; ++i)	// report if this doesn't get unrolled
				y[j] += t_y[i][j];
		}
		for(int i=1; i<SPAWNS; ++i)	// don't delete t_y[0]
			delete [] t_y[i];	
		delete [] t_y;	
		#pragma isat marker SM1_end
	//}
	//else
	//{
	//	A.SeqSpMV(x, y);
	//}
	#pragma isat marker SM2_end
}


// SpMV with symmetric register blocked CSB 
template <typename NT, typename IT, unsigned TTDIM>
void bmsym_gespmv (const BmSym<NT, IT, TTDIM> & A, const NT * __restrict x, NT * __restrict y)
{
	if( A.isPar() )
	{	
		NT * y1 = new NT[A.n]();
		NT * y2 = new NT[A.n]();
		NT * y3;

		IT size0 = A.nrbsum(0);		
		IT size1 = A.nrbsum(1);
		IT size2 = A.nrbsum(2);

		if(size0+size1+size2 != A.nrb)
		{
			y3 = new NT[A.n]();
			cilk_spawn A.MultAddAtomics(y3,x,3);		
		}

		cilk_spawn A.MultDiag(y1,x,1);
		cilk_spawn A.MultDiag(y2,x,2);
		A.MultMainDiag(y, x);

		cilk_sync;

		if(size0+size1+size2 != A.nrb)
		{
			cilk_for(int i=0; i< A.n; ++i)
			{
				y[i] += y1[i] + y2[i] + y3[i];
			}
			delete [] y3;
		}
		else
		{
			cilk_for(int i=0; i< A.n; ++i)
			{
				y[i] += y1[i] + y2[i];
			}
		}
			
		delete [] y1;
		delete [] y2;
	}
	else
	{
		A.SeqSpMV(x, y);
	}
}

// Works on any CSB-like data structure
template <class CSB>
float RowImbalance(const CSB & A)
{
        // get the average without the last left-over blockrow
        float rowave = static_cast<float>(*(A.top[A.nbr-1])) / (A.nbr-1);
        unsigned rowmax = 0;
        for(size_t i=1; i< A.nbr; ++i)
        {
                rowmax = std::max(rowmax, *(A.top[i]) - *(A.top[i-1]));
        }
        return static_cast<float>(rowmax) / rowave;
}


template <class NT, class IT>
float ColImbalance(const BiCsb<NT,IT> & A)
{
        vector<float> sum(A.nbc-1);
        cilk_for(IT j=1; j< A.nbc; ++j)   // ignore the last block column
        {
                IT * blocknnz = new IT[A.nbr];  // nnz per block responsible
                for(IT i=0; i<A.nbr; ++i)
                {
                        blocknnz[i] = A.top[i][j] - A.top[i][j-1];
                }
                sum[j-1] = std::accumulate(blocknnz, blocknnz + (A.nbr-1), 0);         // ignore the last block row
                delete [] blocknnz;
        }
        float colave = std::accumulate(sum.begin(), sum.end(), 0.0) / static_cast<float>(A.nbc-1);
        vector<float>::iterator colmax = std::max_element(sum.begin(), sum.end());
        return (*colmax) / colave;
}


#endif

