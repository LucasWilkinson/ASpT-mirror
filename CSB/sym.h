#ifndef _SYM_H
#define _SYM_H

#include <cmath>
#include "csc.h"
using namespace std;

// Disclaimer: This class heavily uses bitwise operations for indexing. Prefer ITYPE to be "unsigned"
//  The >> operator in C and C++ is not necessarily an arithmetic shift. 
//  Usually it is only an arithmetic shift if used on a signed integer type; 
//  but if it is used on an unsigned integer type, it will be a logical shift

template <class T, class ITYPE>
class Sym
{
public:
	Sym ():nz(0), m(0), n(0), ntop(0), nbc(0), nbr(0) {}	// default constructor (dummy)

	Sym (ITYPE size,ITYPE rows, ITYPE cols);
	Sym (const Sym<T, ITYPE> & rhs);			// copy constructor
	~Sym();
	Sym<T,ITYPE> & operator=(const Sym<T,ITYPE> & rhs);	// assignment operator
	Sym (Csc<T, ITYPE> & csc);

	ITYPE colsize() const { return n;} 
	ITYPE rowsize() const { return m;} 
	ITYPE getntop() const { return ntop; }

	void Transpose();

private:
	void Init();
	void SubSpMV(ITYPE * btop, ITYPE bstart, ITYPE bend, const T * x, T * suby) const;
	void BMult(ITYPE * btop, ITYPE bstart, ITYPE bend, const T * x, T * y, ITYPE ysize) const;
	static void BlockPrefetch(void * addr,  int total, int ssize);

	ITYPE ** top ;	// pointers array (indexed by higher-order bits of the coordinate index), size ~= ntop+1
	ITYPE * bot;	// contains lower-order bits of the coordinate index, size nnz 
	T * num;		// contains numerical values, size nnz
	
	ITYPE nz;		// # nonzeros
	ITYPE m;		// # rows
	ITYPE n;		// # columns
	ITYPE ntop;		// size of top array ~= sqrt(mn)

	ITYPE nbc;		// #{column blocks} = #{blocks in any block row}
	ITYPE nbr; 		// #{block rows)
	
	ITYPE rowlowbits;	// # lower order bits for rows
	ITYPE rowhighbits;
	ITYPE highrowmask;  // mask with the first log(m)/2 bits = 1 and the other bits = 0  
	ITYPE lowrowmask;	

	ITYPE collowbits;	// # lower order bits for columns
	ITYPE colhighbits;
	ITYPE highcolmask;  // mask with the first log(n)/2 bits = 1 and the other bits = 0  
	ITYPE lowcolmask;

	static const int CACHEBLOCK = 16; // block prefetch blocks of 16 ints/doubles 
	static int p_fetch;		// anchor variable to fool the C optimizer

	template <typename U, typename UTYPE>
	friend void sym_gaxpy (const Sym<U, UTYPE> & A, const U * x, U * y);
};


// const int* pX;    --> changeable pointer to constant int
// int* const pY;    --> constant pointer to changeable int

// Operation y = A*x+y 
template <typename T, typename ITYPE>
void sym_gaxpy (const Sym<T, ITYPE> & A, const T * x, T * y)
{
	// Some bitwise algebra for reference
	// If b = (a & mask) >> shift, then how to recover a back?
	// Easy...  a = (b << shift) | (~mask)

	ITYPE ysize = A.lowrowmask + 1;				// size of the output subarray (per block row)

	cilk_for (ITYPE i = 0 ; i < A.nbr ; ++i)	// for all blocks rows of A 
	{
		ITYPE * btop = A.top [i];				// get the pointer to this block row
		ITYPE rhi = ((i << A.rowlowbits) & A.highrowmask);
		T * suby = &y[rhi];

#ifdef PAR2D
		A.BMult(btop, 0, A.nbc, x, suby, ysize);
#else
		A.SubSpMV(btop, 0, A.nbc, x, suby);
#endif
    }
}


#include "sym.cpp"	// Template member function definitions need to be known to the compiler
#endif
