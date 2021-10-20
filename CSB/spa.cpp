#include "spa.h"
#include "utility.h"
#include <cassert>


// copy constructor
template <class T, class ITYPE>
Spa<T, ITYPE>::Spa (const Spa<T,ITYPE> & rhs)
: length(rhs.length)
{
	if(length > 0)
	{
		indices = rhs.indices;
		bitmap = new bool[length];
		values = new T[length];
		
		for(ITYPE i=0; i< length; ++i)	
			bitmap[i]= rhs.bitmap[i];
		for(ITYPE i=0; i< length; ++i)	
			values[i]= rhs.values[i];
	}
}

template <class T, class ITYPE>
Spa<T, ITYPE> & Spa<T, ITYPE>::operator= (const Spa<T, ITYPE> & rhs)
{
	if(this != &rhs)		
	{
		if(length > 0)	// if the existing object is not empty, make it empty
		{
			delete [] bitmap;
			delete [] values;
		}
		length = rhs.length;

		if(length > 0)
		{
			indices = rhs.indices;
			bitmap = new bool[length];
			values = new T[length];
		
			for(ITYPE i=0; i< length; ++i)	
				bitmap[i]= rhs.bitmap[i];
			for(ITYPE i=0; i< length; ++i)	
				values[i]= rhs.values[i];
		}
	}
	return *this;
}

template <class T, class ITYPE>
Spa<T, ITYPE>::~Spa()
{
	if( length > 0)
	{
		delete [] bitmap;
		delete [] bitmap;
	}
}

template <class T, class ITYPE>
Spa<T, ITYPE>::Scatter(T value, ITYPE pos)
{
	if(bitmap[pos])
		values[pos] += value;
	else
	{
		bitmap[pos] = true;
		values[pos] = value;
		indices.push_back(pos);
	}
}

template <class T, class ITYPE>
Spa<T, ITYPE>::Gather(T * y)
{
	ITYPE nnz = indices.size();
	for(ITYPE =0; i<nnz; ++i)
	{
		y[indices[i]] += values[indices[i]];
		values[indices[i]] = 0;
		bitmap[indices[i]] = false;
	}
	indices.clear();	// free vector, and set size() to 0
}