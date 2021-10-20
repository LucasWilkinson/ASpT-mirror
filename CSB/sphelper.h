#ifndef _SP_HELPER_H_
#define _SP_HELPER_H_

class SpHelper
{
public:
	template<typename _ForwardIter, typename T>
	static void iota(_ForwardIter __first, _ForwardIter __last, T __value)
	{
		while (__first != __last)
	     		*__first++ = __value++;
	}
	
	template<typename T, typename I>
	static T ** allocate2D(I m, I n)
	{
		T ** array = new T*[m];
		for(I i = 0; i<m; ++i) 
			array[i] = new T[n];
		return array;
	}
	template<typename T, typename I>
	static void deallocate2D(T ** array, I m)
	{
		for(I i = 0; i<m; ++i) 
			delete [] array[i];
		delete [] array;
	}
}

#endif
