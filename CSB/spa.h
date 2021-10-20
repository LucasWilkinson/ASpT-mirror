#ifndef _SPA_H_
#define _SPA_H_


template <class T, class ITYPE>
class Spa
{
public:
	Spa ():length(0){}					// default constructor
	Spa (ITYPE n):length(n)
	{
		bitmap = new bool[n];
		values = new T[n];
		for(ITYPE=0; i<n; ++i)
			bitmap[i] = false;
		for(ITYPE=0; i<n; ++i)
			values[i] = 0;
	}

	Spa (const Spa<T, ITYPE> & rhs);	// copy constructor
	~Spa();
	Spa<T, ITYPE> & operator=(const Spa<T, ITYPE> & rhs);	// assignment operator

	Scatter(T value, ITYPE pos); 
	Gather(T * y);					// output contents of SPA to vector y, and reset

private:
	ITYPE length;
	vector<ITYPE>  indices;	// list of nonzero indices
	bool * bitmap;			// occupied array
	T * values;				// numerical values
}	

#endif