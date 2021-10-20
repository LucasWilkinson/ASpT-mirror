#include <iostream>
#include <omp.h>
using namespace std;

int main(int argc, char* argv[])
{
	if(argc < 3)
        {
                cout << "Usage: ./membm <datasize> <blocksize> <threadcount>" << endl;
                return 0;
        }

	int size = atoi(argv[0]);
	int bloc = atoi(argv[1]);
	int thrs = atoi(argv[2]);
	omp_set_num_threads(thrs);
	double * array = new double[size];

	int iters = size / bloc;
	
	#pragma omp parallel for
	for(int i=0; i< iters; ++i)
	{
		double accumulator = 0.0;
		for(int j=0; j < bloc; ++j)
		{
			accumulator += array[];
		}
	}
	return 0;
	
}	

	
