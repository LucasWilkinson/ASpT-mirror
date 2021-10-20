#define NOMINMAX
#include <iostream> 
#include <algorithm>
#include <functional>
#include <fstream>
#include <ctime>
#include <cmath>
#include <string>

#include "cilk_util.h"
#include "utility.h"

#include "triple.h"
#include "csc.h"
#include "bicsb.h"
#include "spvec.h"

using namespace std;

#define INDEXTYPE uint32_t
#define VALUETYPE double

/* Alternative native timer (wall-clock):
 *	timeval tim;		
 *	gettimeofday(&tim, NULL);
 *	double t1=tim.tv_sec+(tim.tv_usec/1000000.0);
 */

INDEXTYPE flops;

int main(int argc, char* argv[])
{
#ifndef	CILK_STUB
	int gl_nworkers = __cilkrts_get_nworkers();
#else
	int gl_nworkers = 0;
#endif
	bool syminput = false;
	bool binary = false;
	bool iscsc = false;
	INDEXTYPE m = 0, n = 0, nnz = 0, forcelogbeta = 0;
	string inputname;
	if(argc < 2)
	{
		cout << "Normal usage: ./parspmvt inputmatrix.mtx sym/nosym binary/text triples/csc" << endl;
		cout << "Assuming matrix.txt is the input, matrix is unsymmetric, and stored in text(ascii) file" << endl;
		inputname = "matrix.txt";
	}
	else if(argc < 3)
	{
		cout << "Normal usage: ./parspmvt inputmatrix.mtx sym/nosym binary/text triples/csc" << endl;
		cout << "Assuming that the matrix is unsymmetric, and stored in text(ascii) file" << endl;
		inputname =  argv[1];
	}
	else if(argc < 4)
	{
		cout << "Normal usage: ./parspmvt inputmatrix.mtx sym/nosym binary/text triples/csc" << endl;
		cout << "Assuming matrix is stored in text(ascii) file" << endl;
		inputname =  argv[1];
		string issym(argv[2]);
		if(issym == "sym")
			syminput = true;
		else if(issym == "nosym")
			syminput = false;
		else
			cout << "unrecognized option, assuming nosym" << endl;
	}
	else
	{
		inputname =  argv[1];
		string issym(argv[2]);
		if(issym == "sym")
			syminput = true;
		else if(issym == "nosym")
			syminput = false;
		else
			cout << "unrecognized option, assuming unsymmetric" << endl;

		string isbinary(argv[3]);
		if(isbinary == "text")
			binary = false;
		else if(isbinary == "binary")
			binary = true;
		else
			cout << "unrecognized option, assuming text file" << endl;

		if(argc > 4)
		{
			string type(argv[4]);
			if(type == "csc")
			{
				iscsc = true;
				cout << "Processing CSC binary" << endl;
			}
		}
			
		if(argc == 6)
			forcelogbeta = atoi(argv[4]);
	}
	typedef PTSR<VALUETYPE,VALUETYPE> PTDD;		
	if(binary)
	{
		Csc<VALUETYPE, INDEXTYPE> * csc;
		FILE * f = fopen(inputname.c_str(), "r");
		if(!f)
		{
			cerr << "Problem reading binary input file\n";
			return 1;
		}
		if(iscsc)
		{
			fread(&n, sizeof(INDEXTYPE), 1, f);
                        fread(&m, sizeof(INDEXTYPE), 1, f);
                        fread(&nnz, sizeof(INDEXTYPE), 1, f);
		}
		else
		{
			fread(&m, sizeof(INDEXTYPE), 1, f);
			fread(&n, sizeof(INDEXTYPE), 1, f);
			fread(&nnz, sizeof(INDEXTYPE), 1, f);
		}
		if (m <= 0 || n <= 0 || nnz <= 0)
		{
			cerr << "Problem with matrix size in binary input file\n";	
			return 1;		
		}
		long tstart = cilk_get_time();	// start timer
		cout << "Reading matrix with dimensions: "<< m << "-by-" << n <<" having "<< nnz << " nonzeros" << endl;
		INDEXTYPE * rowindices = new INDEXTYPE[nnz];
		VALUETYPE * vals = new VALUETYPE[nnz];
		INDEXTYPE * colindices;
		INDEXTYPE * colpointers;
		if(iscsc)
		{
			colpointers = new INDEXTYPE[n+1];
			size_t cols = fread(colpointers, sizeof(INDEXTYPE), n+1, f);
			if(cols != n+1)
			{
				cerr << "Problem with FREAD, aborting... " << endl;
                        	return -1;
			}
		}
		else
		{
			colindices = new INDEXTYPE[nnz];
			size_t cols = fread(colindices, sizeof(INDEXTYPE), nnz, f);
			if(cols != nnz)
			{
				cerr << "Problem with FREAD, aborting... " << endl;
                        	return -1;
			}
		}
		size_t rows = fread(rowindices, sizeof(INDEXTYPE), nnz, f);
		size_t nums = fread(vals, sizeof(VALUETYPE), nnz, f);

		if(rows != nnz || nums != nnz)
		{
			cerr << "Problem with FREAD, aborting... " << endl;
			return -1;
		}
		long tend = cilk_get_time();	// end timer	
		cout<< "Reading matrix in binary took " << ((VALUETYPE) (tend-tstart)) /1000 << " seconds" <<endl;
		fclose(f);
		if(iscsc)
		{
			csc = new Csc<VALUETYPE, INDEXTYPE>();
			csc->SetPointers(colpointers, rowindices, vals , nnz, m, n, true);	// do the fortran thing
			// csc itself will manage the data in this case (shallow copy)
		}
		else
		{
			csc = new Csc<VALUETYPE, INDEXTYPE>(rowindices, colindices, vals , nnz, m, n);
			delete [] colindices;
			delete [] rowindices;
			delete [] vals;
		}

	        BiCsb<VALUETYPE, INDEXTYPE> bicsb(*csc, gl_nworkers);
	
		flops = 2 * nnz;

		cout << "# workers: "<< gl_nworkers << endl;
		cout << "generating vectors... " << endl;
		cout << "Starting SpMV_T..." << endl;		
		cout << "Col imbalance is: " << ColImbalance(bicsb) << endl;


		Spvec<VALUETYPE, INDEXTYPE> xt(m);
		Spvec<VALUETYPE, INDEXTYPE> yt_bicsb(n);
		Spvec<VALUETYPE, INDEXTYPE> yt_csc (n);
		yt_csc.fillzero();
		yt_bicsb.fillzero();
		xt.fillrandom();
	
		bicsb_gespmvt<PTDD>(bicsb, xt.getarr(), yt_bicsb.getarr());		// dummy computation
	
		long t0 = cilk_get_time();	// start timer
		for(int i=0; i < REPEAT; ++i)
		{
			bicsb_gespmvt<PTDD>(bicsb, xt.getarr(), yt_bicsb.getarr());
		}
		long t1 = cilk_get_time();	// get the wall-clock time

		long time = (t1-t0)/REPEAT;
		cout<< "BiCSB Trans" << " time: " << ((double) time) /1000 << " seconds" <<endl;
		cout<< "BiCSB Trans" << " mflop/sec: " << flops / (1000 * (double) time) <<endl;
		
		csc_gaxpy_trans ( *csc, xt.getarr(), yt_csc.getarr());
	        t0 = cilk_get_time();
        	for(int i=0; i < REPEAT; ++i)
        	{
                	csc_gaxpy_trans ( *csc, xt.getarr(), yt_csc.getarr());
        	}
        	t1 = cilk_get_time();
        	time = (t1-t0)/REPEAT;
        	cout <<"Transposed CSC time: " << ((double) time) / 1000 << " seconds" << endl;
	        cout <<"Transposed CSC mflop/sec: " << flops/ (1000 * (double) time) << endl;

		Verify(yt_csc, yt_bicsb, "BiCSB", n);	// inside Spvec.cpp
#ifdef STATS
		ofstream stats("stats.txt");
		bicsb.PrintStats(stats);
		stats.close();
#endif
	}
	else
	{
		cout << "reading input matrix in text(ascii)... " << endl;
		ifstream infile(inputname.c_str());
		char line[256];
		char c = infile.get();
		while(c == '%')
		{
			infile.getline(line,256);
			c = infile.get();
		}
		infile.unget();
		infile >> m >> n >> nnz;	// #{rows}-#{cols}-#{nonzeros}
		flops = 2*nnz;	

		long tstart = cilk_get_time();	// start timer	
		Triple<VALUETYPE, INDEXTYPE> * triples = new Triple<VALUETYPE, INDEXTYPE>[nnz];
	
		if (infile.is_open())
		{
			INDEXTYPE cnz = 0;	// current number of nonzeros
			while (! infile.eof() && cnz < nnz)
			{
				infile >> triples[cnz].row >> triples[cnz].col >> triples[cnz].val;	// row-col-value
				triples[cnz].row--;
				triples[cnz].col--;
				++cnz;
			}
			assert(cnz == nnz);	
		}
		long tend = cilk_get_time();	// end timer	
		cout<< "Reading matrix in ascii took " << ((double) (tend-tstart)) /1000 << " seconds" <<endl;
	
		cout << "converting to csc and bicsb... " << endl;
		Csc<VALUETYPE, INDEXTYPE> csc(triples, nnz, m, n);
		delete [] triples;

		BiCsb<VALUETYPE, INDEXTYPE> bicsb(csc, gl_nworkers);
		cout << "# workers: "<< gl_nworkers << endl;
		cout << "generating vectors... " << endl;
		cout << "starting matvecs... " << endl;
	
		Spvec<VALUETYPE, INDEXTYPE> xt(m);
		Spvec<VALUETYPE, INDEXTYPE> yt_csc(n);
		Spvec<VALUETYPE, INDEXTYPE> yt_bicsb(n);
		yt_csc.fillzero();
		yt_bicsb.fillzero();
		xt.fillrandom();
	
		csc_gaxpy_trans(csc, xt.getarr(), yt_csc.getarr());		// dummy computation
	
		long t0 = cilk_get_time();	// start timer
		for(int i=0; i < REPEAT; ++i)
		{
			csc_gaxpy_trans(csc, xt.getarr(), yt_csc.getarr());
		}
		long t1 = cilk_get_time();	// get the wall-clock time

		long time = (t1-t0)/REPEAT;
		cout<< "CSC Trans" << " time: " << ((double) time) /1000 << " seconds" <<endl;
		cout<< "CSC Trans" << " mflop/sec: " << flops / (1000 * (double) time) <<endl;
	
		bicsb_gespmvt<PTDD>(bicsb, xt.getarr(), yt_bicsb.getarr());		// dummy computation

		t0 = cilk_get_time();	// start timer
		for(int i=0; i < REPEAT; ++i)
		{
			bicsb_gespmvt<PTDD>(bicsb, xt.getarr(), yt_bicsb.getarr());
		}
		t1 = cilk_get_time();	// get the wall-clock time

		time = (t1-t0)/REPEAT;
		cout<< "BiCSB Trans" << " time: " << ((double) time) /1000 << " seconds" <<endl;
		cout<< "BiCSB Trans" << " mflop/sec: " << flops / (1000 * (double) time) <<endl;
	
		Verify(yt_csc, yt_bicsb, "BiCSB", n);	// inside Spvec.cpp
		return 0;
	}	
}

