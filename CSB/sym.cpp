#include "sym.h"
#include "utility.h"
#include <cassert>


// You must declare "static data members" inside the class body, 
// and you also need to define them outside the class body. 
// When the program refers to a static data member that is only declared but not defined, 
// the "unresolved external symbol" error occurs.
template <class T, class ITYPE>
int Sym<T, ITYPE>::p_fetch = 0;


template <class T, class ITYPE>
void Sym<T, ITYPE>::Init()
{
	ITYPE roundrowup = nextpoweroftwo(m);
	ITYPE roundcolup = nextpoweroftwo(n);
	ITYPE roundnnzup = nextpoweroftwo(nz);

	ITYPE rowbits = highestbitset(roundrowup);
	ITYPE colbits = highestbitset(roundcolup);
	ITYPE nnzbits = highestbitset(roundnnzup);

	if(!(rowbits > 1 && colbits > 1))
	{
		cerr << "Matrix too small for this library" << endl;
		return;
	}

	rowhighbits = static_cast<ITYPE>(rowbits/2)-1;	// # higher order bits for rows
	colhighbits = static_cast<ITYPE>(colbits/2)-1;	// # higher order bits for columns

	rowlowbits = rowbits - rowhighbits;		// rowlowbits = # lower order bits for rows
	collowbits = colbits - colhighbits;		// collowbits = # lower order bits for columns

	lowrowmask = IntPower<ITYPE>(2,rowlowbits) - 1;
	highrowmask = ((roundrowup - 1) ^ lowrowmask);
	lowcolmask = IntPower<ITYPE>(2,collowbits) - 1;
	highcolmask = ((roundcolup - 1) ^ lowcolmask);

	nbc = IntPower<ITYPE>(2,colhighbits);		// #{block columns} = #{blocks in any block row}
	nbr = IntPower<ITYPE>(2,rowhighbits);		// #{block rows)
	ntop = nbc * nbr;							// size of top array
}

// Constructing empty Sym objects (size = 0) are not allowed.
template <class T, class ITYPE>
Sym<T, ITYPE>::Sym (ITYPE size, ITYPE rows, ITYPE cols): nz(size),m(rows),n(cols)
{
	assert(nz != 0 && n != 0 && m != 0);
	Init();

	num = new T[nz];
	bot = new ITYPE[nz];
	top = new ITYPE* [nbr];	

	for(ITYPE i=0; i<nbr; ++i)
		top[i] = new ITYPE[nbc+1]; 
}


// copy constructor
template <class T, class ITYPE>
Sym<T, ITYPE>::Sym (const Sym<T,ITYPE> & rhs)
: nz(rhs.nz), m(rhs.m), n(rhs.n), ntop(rhs.ntop), nbr(rhs.nbr), nbc(rhs.nbc), rowhighbits(rhs.rowhighbits),rowlowbits(rhs.rowlowbits),
highrowmask(rhs.highrowmask), lowrowmask(rhs.lowrowmask), colhighbits(rhs.colhighbits), collowbits(rhs.collowbits),
highcolmask(rhs.highcolmask), lowcolmask(rhs.lowcolmask)
{
	if(nz > 0)
	{
		num = new T[nz];
		bot = new ITYPE[nz];

		for(ITYPE i=0; i< nz; ++i)	
			num[i]= rhs.num[i];
		for(ITYPE i=0; i< nz; ++i)	
			bot[i]= rhs.bot[i];
	}
	if ( ntop > 0)
	{
		top = new ITYPE* [nbr];

		for(ITYPE i=0; i<nbr; ++i)
			top[i] = new ITYPE[nbc+1]; 

		for(ITYPE i=0; i<nbr; ++i)
			for(ITYPE j=0; j <= nbc; ++j) 
				top[i][j] = rhs.top[i][j];
	}
}

template <class T, class ITYPE>
Sym<T, ITYPE> & Sym<T, ITYPE>::operator= (const Sym<T, ITYPE> & rhs)
{
	if(this != &rhs)		
	{
		if(nz > 0)	// if the existing object is not empty
		{
			// make it empty
			delete [] bot;
			delete [] num;
		}
		if(ntop > 0)
		{
			for(ITYPE i=0; i<nbr; ++i)
				delete [] top[i];
			delete [] top;
		}

		nz	= rhs.nz;
		n	= rhs.n;
		m   	= rhs.m;
		ntop	= rhs.ntop;
		nbr 	= rhs.nbr;
		nbc 	= rhs.nbc;

		rowhighbits = rhs.rowhighbits;
		rowlowbits	= rhs.rowlowbits;
		highrowmask = rhs.highrowmask;
		lowrowmask	= rhs.lowrowmask;

		colhighbits = rhs.colhighbits;
		collowbits = rhs.collowbits;
		highcolmask = rhs.highcolmask;
		lowcolmask	= rhs.lowcolmask;

		if(nz > 0)	// if the copied object is not empty
		{
			num = new T[nz];
			bot = new ITYPE[nz];

			for(ITYPE i=0; i< nz; ++i)	
				num[i]= rhs.num[i];
			for(ITYPE i=0; i< nz; ++i)	
				bot[i]= rhs.bot[i];
		}
		if(ntop > 0)
		{
			top = new ITYPE* [nbr];

			for(ITYPE i=0; i<nbr; ++i)
				top[i] = new ITYPE[nbc+1]; 

			for(ITYPE i=0; i<nbr; ++i)
				for(ITYPE j=0; j <= nbc; ++j) 
					top[i][j] = rhs.top[i][j];
		}
	}
	return *this;
}

template <class T, class ITYPE>
Sym<T, ITYPE>::~Sym()
{
	if( nz > 0)
	{
		delete [] bot;
		delete [] num;
	}
	if ( ntop > 0)
	{
		for(ITYPE i=0; i<nbr; ++i)
			delete [] top[i];
		delete [] top;
	}
}


template <class T, class ITYPE>
Sym<T, ITYPE>::Sym (Csc<T, ITYPE> & csc):nz(csc.nz),m(csc.m),n(csc.n)
{
	typedef std::pair<ITYPE, ITYPE> ipair;
	typedef std::pair<ITYPE, ipair> mypair;

	assert(nz != 0 && n != 0 && m != 0);
	Init();

	num = new T[nz];
	bot = new ITYPE[nz];
	
	top = new ITYPE* [nbr];
	for(ITYPE i=0; i<nbr; ++i)
		top[i] = new ITYPE[nbc+1]; 

	mypair * pairarray = new mypair[nz];
	ITYPE k = 0;
	for(ITYPE j = 0; j < n; ++j)
	{
		for (ITYPE i = csc.jc [j] ; i < csc.jc[j+1] ; ++i)	// scan the jth column
		{
			// concatenate the higher/lower order half of both row (first) index and col (second) index bits 
			ITYPE hindex = (((highrowmask &  csc.ir[i] ) >> rowlowbits) << colhighbits)
										| ((highcolmask & j) >> collowbits);	
			ITYPE lindex = ((lowrowmask &  csc.ir[i]) << collowbits) | (lowcolmask & j) ;

			// i => location of that nonzero in csc.ir and csc.num arrays
			pairarray[k++] = mypair(hindex, ipair(lindex,i));
		}
	}
	sort(pairarray, pairarray+nz);	// sort according to hindex

	// Now, elements within each block are sorted with respect to their lindex (i.e. there are in row-major order)
	// This is because the default comparison operator of pair<T1,T2> uses lexicographic comparison: 
	// within each block, hindex is the same, so they are sorted w.r.t ipair(lindex,i) lexicographically.

	ITYPE cnz = 0;

	for(ITYPE i = 0; i < nbr; ++i)
	{
		for(ITYPE j = 0; j < nbc; ++j)
		{
			top[i][j] = cnz;
			while(cnz < nz && pairarray[cnz].first == ((i*nbc)+j) )	// as long as we're in that block
			{
				bot[cnz] = (pairarray[cnz].second).first;
				num[cnz++] = csc.num[(pairarray[cnz].second).second];
			}
		}
		top[i][nbc] = cnz;
	}
		
	delete [] pairarray;
}

template <typename T, typename ITYPE>
void Sym<T, ITYPE>::BMult(ITYPE * btop, ITYPE bstart, ITYPE bend, const T * x, T * y, ITYPE ysize) const
{
	if((btop[bend] - btop[bstart] < BREAKEVEN * ysize) || (bend-bstart == 1))	
	{
		// not enough nonzeros to amortize new array formation
		SubSpMV(btop, bstart, bend, x, y);
		return;
	}
	ITYPE nnb = (bend+bstart)/2;	

	cilk_spawn BMult(btop, bstart, nnb, x, y, ysize);
	if(SYNCHED)
	{
		BMult(btop, nnb, bend, x, y, ysize);
	}
	else
	{
		T * temp = new T[ysize];
		for(ITYPE i=0; i<ysize; ++i)
			temp[i] = 0.0;

		BMult(btop, nnb, bend, x, temp, ysize);
		cilk_sync;
		
		for(ITYPE i=0; i<ysize; ++i)
			y[i] += temp[i];

		delete [] temp;
	}
}

// double* restrict a; --> No aliases for a[0], a[1], ...
// bstart: block start index
// bend: block end index
template <class T, class ITYPE>
inline void Sym<T, ITYPE>::SubSpMV(ITYPE * __restrict btop, ITYPE bstart, ITYPE bend, 
							const T * __restrict x, T * __restrict suby) const
{
	for (ITYPE j = bstart ; j < bend ; ++j)		// for all blocks inside that block row
	{
		// get higher order bits for column indices
		ITYPE chi = ((j << collowbits) & highcolmask);
		const T * subx = &x[chi];
 
#ifdef AMDPREOPT
		//BlockPrefetch(&bot[btop[j]], btop[j+1]-btop[j], sizeof(ITYPE));
		//BlockPrefetch(&num[btop[j]], btop[j+1]-btop[j], sizeof(T));
		// optimization: block prefetch
		
		ITYPE blocknz = btop[j+1] - btop[j];
		if(blocknz > 8 && blocknz < 4096)
		{
			// number of elements in one cache line
			const int botinc = CLSIZE/sizeof(ITYPE);	
			const int numinc = CLSIZE/sizeof(T);

			for (ITYPE i=0; i < blocknz; i+=botinc)				// prefetch once for every cache line
			{
				__builtin_prefetch (&bot[btop[j]+i], 0, 0);		// prefetch read-only, with no temporal locality
			}
			for (ITYPE i=0; i < blocknz; i+=numinc)
			{	
				__builtin_prefetch (&num[btop[j]+i], 0, 0);	
			}
		}
#endif

#ifdef SSEOPT
		__m128i m1 = _mm_set_epi32 (mu1, mu1, mu1, mu1);
		__m128i m2 = _mm_set_epi64x(mlong1, mlong2);	// pack two __int64 integers (for x86_64)
#endif

#ifdef PERFDEB
		ofstream stat("Perfdeb.txt", ios::app);
		stat << btop[j+1] - btop[j] << endl;
#endif

		for (ITYPE k = btop[j] ; k < btop[j+1] ; ++k)	// for all nonzeros within ith block (expected =~ nnz/n = c)
		{			
			ITYPE rli = ((bot[k] >> collowbits) & lowrowmask);
			ITYPE cli = (bot[k] & lowcolmask);

			suby [rli] += num[k] * subx [cli] ;
		}
	}
}

// This is a static member function does not have a "this" pointer
template <class T, class ITYPE>
void Sym<T, ITYPE>::BlockPrefetch(void * addr,  int total, int ssize)
{
	p_fetch = 0;
	int * __restrict a = (int*) addr;
	const int inc = CLSIZE/ssize;	// number of elements in one cache line

	// Grab every 64th address
	for(int i=0; i<total; i+=inc)
	{
		p_fetch += a[i];
	}
}



template <class T, class ITYPE>
void Sym<T, ITYPE>::Transpose()
{
	// when we jump to the next block in the same block-column, we move leaddim positions inside "top" array
	// leadim ~= sqrt(n) => number of blocks in each block-row
	ITYPE leaddim = lowcolmask+1;	
	Sym symT(nz, m, n);				// create empty transposed object

	ITYPE k = 0;
	ITYPE cnz = 0;

	for(ITYPE j = 0; j < leaddim; ++j)	// scan columns of top-level structure (~sqrt(n) iterations)
	{
		for(ITYPE i = j; i < ntop ; i += leaddim)	// iterates ~ sqrt(m) times within the block column
		{
			symT.top[k++] = cnz;
			cnz += top[i+1]-top[i];
		}
	}
	symT.top[k] = cnz;

	// Embarrassingly parallel sort of indices to get new bottom array
	// ITYPE nindex = (highmask &  csc.ir [i]) | ((highmask & bot) >> 4);
}

