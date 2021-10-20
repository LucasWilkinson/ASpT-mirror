#define NOMINMAX
#include <iostream> 
#include <algorithm>
#include <numeric>
#include <functional>
#include <fstream>
#include <ctime>
#include <cmath>
#include <string>

#include "utility"
#include "timer.gettimeofday.c"
#include "cilk_util.h"

#include "triple.h"
#include "csc.h"
#include "csbsym.h"
#include "bmsym.h"
#include "spvec.h"
#include "Semirings.h"

using namespace std;

#define INDEXTYPE unsigned
#define VALUETYPE double

/* Alternative native timer (wall-clock):
 *	timeval tim;		
 *	gettimeofday(&tim, NULL);
 *	double t1=tim.tv_sec+(tim.tv_usec/1000000.0);
 */


int main(int argc, char* argv[])
{
#ifndef CILK_STUB
	int gl_nworkers = WORKERS;
#else
	int gl_nworkers = 0;
#endif
	bool syminput = true;
	bool binary = false;
	INDEXTYPE m = 0, n = 0, nnz = 0, forcelogbeta = 0;
	string inputname;
	if(argc < 2)
	{
		cout << "Normal usage: ./a.out inputmatrix.mtx sym/nosym binary/text" << endl;
		cout << "Assuming matrix.txt is the input, matrix is symmetric, and stored in text(ascii) file" << endl;
		inputname = "matrix.txt";
	}
	else if(argc < 3)
	{
		cout << "Normal usage: ./a.out inputmatrix.mtx sym/nosym binary/text" << endl;
		cout << "Assuming that the matrix is symmetric, and stored in text(ascii) file" << endl;
		inputname =  argv[1];
	}
	else if(argc < 4)
	{
		cout << "Normal usage: ./a.out inputmatrix.mtx sym/nosym binary/text" << endl;
		cout << "Assuming matrix is stored in text(ascii) file" << endl;
		inputname =  argv[1];
		string issym(argv[2]);
		if(issym == "sym")
			syminput = true;
		else if(issym == "nosym")
			syminput = false;
		else
			cout << "unrecognized option, assuming sym" << endl;
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
			cout << "unrecognized option, assuming symmetric" << endl;

		string isbinary(argv[3]);
		if(isbinary == "text")
			binary = false;
		else if(isbinary == "binary")
			binary = true;
		else
			cout << "unrecognized option, assuming text file" << endl;
	
		if(argc == 5)
			forcelogbeta = atoi(argv[4]);
	}

	Csc<VALUETYPE, INDEXTYPE> * csc;
	Csc<VALUETYPE, INDEXTYPE> * cscsym;
	if(binary)
	{
		FILE * f = fopen(inputname.c_str(), "r");
		if(!f)
		{
			cerr << "Problem reading binary input file\n";
			return 1;
		}
		fread(&m, sizeof(INDEXTYPE), 1, f);
		fread(&n, sizeof(INDEXTYPE), 1, f);
		fread(&nnz, sizeof(INDEXTYPE), 1, f);

		if (m <= 0 || n <= 0 || nnz <= 0)
		{
			cerr << "Problem with matrix size in binary input file\n";	
			return 1;		
		}

		long tstart = cilk_get_time();	// start timer
		cout << "Reading matrix with dimensions: "<< m << "-by-" << n <<" having "<< nnz << " nonzeros" << endl;
		
		INDEXTYPE * rowindices = new INDEXTYPE[nnz];
		INDEXTYPE * colindices = new INDEXTYPE[nnz];
		VALUETYPE * vals = new VALUETYPE[nnz];

		size_t rows = fread(rowindices, sizeof(INDEXTYPE), nnz, f);
		size_t cols = fread(colindices, sizeof(INDEXTYPE), nnz, f);
		size_t nums = fread(vals, sizeof(VALUETYPE), nnz, f);

		if(rows != nnz || cols != nnz || nums != nnz)
		{
			cerr << "Problem with FREAD, aborting... " << endl;
			return -1;
		}

		long tend = cilk_get_time();	// end timer	
		cout<< "Reading matrix in binary took " << ((VALUETYPE) (tend-tstart)) /1000 << " seconds" <<endl;
		fclose(f);
		
		cscsym = new Csc<VALUETYPE, INDEXTYPE>(rowindices, colindices, vals , nnz, m, n, true);	// create symmetric csc
		csc = new Csc<VALUETYPE, INDEXTYPE>(rowindices, colindices, vals , nnz, m, n);	// create unsymmetric csc

		delete [] rowindices;
		delete [] colindices;
		delete [] vals;
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
	
		cout << "converting to csc ... " << endl;
		cscsym = new Csc<VALUETYPE,INDEXTYPE>(triples, nnz, m, n,true);	// create symmetric csb
		csc = new Csc<VALUETYPE,INDEXTYPE>(triples, nnz, m, n);	// create unsymmetric csb
		delete [] triples;
	}

	CsbSym<VALUETYPE, INDEXTYPE> csbsym(*cscsym, gl_nworkers);
#ifndef NOBM
	BmSym<VALUETYPE, INDEXTYPE, RBDIM> bmsym(*cscsym, gl_nworkers);
#endif

	cout << "# workers: "<< gl_nworkers << endl;
		
	ofstream stats("stats.txt");
#ifdef  DEBUG
	csbym.Dump(stats);
#else	
	csbsym.PrintStats(stats);
#endif
		
	INDEXTYPE flops = 2 * cscsym->getlogicalnnz();
	cout << "generating vectors... " << endl;
	Spvec<VALUETYPE, INDEXTYPE> x(n);
	Spvec<VALUETYPE, INDEXTYPE> y_csbsym(m);
	Spvec<VALUETYPE, INDEXTYPE> y_bmsym(m);
	Spvec<VALUETYPE, INDEXTYPE> y_cscsym(m);
	Spvec<VALUETYPE, INDEXTYPE> y_csc(m);

	y_csbsym.fillzero();
	y_bmsym.fillzero();
	y_cscsym.fillzero();
	y_csc.fillzero();
	x.fillfota();
	
	timer_init();
	cout << "starting SpMV ... " << endl;

#ifdef STATS
	cilk::cilkview cv;
	cv.start();
#endif
	csbsym_gespmv(csbsym, x.getarr(), y_csbsym.getarr());
#ifdef STATS
	cv.stop();
	cv.dump("sym_spmv");
	cout << "Total flops: " << flops << ", atomic flops: " << atomicflops.get_value() << ", ratio: " << static_cast<float>(atomicflops.get_value()) / flops << endl;
#endif

	double t0 = timer_seconds_since_init();
	for(int i=0; i < REPEAT; ++i)
	{
		csbsym_gespmv(csbsym, x.getarr(), y_csbsym.getarr());
	}
	double t1 = timer_seconds_since_init();

	double time = (t1-t0)/REPEAT;
	cout<< "CsbSym" << " time: " << time << " seconds" <<endl;
	cout<< "CsbSym" << " mflop/sec: " << (flops  / (1000000 * time)) <<endl;

	//-----------------------------------------//
#ifndef NOBM	
	bmsym_gespmv(bmsym, x.getarr(), y_bmsym.getarr());
	t0 = timer_seconds_since_init();
	for(int i=0; i < REPEAT; ++i)
	{
		bmsym_gespmv(bmsym, x.getarr(), y_bmsym.getarr());
	}
	t1 = timer_seconds_since_init();
	time = (t1-t0)/REPEAT;
	cout<< "BmSym" << " time: " << time << " seconds" <<endl;
	cout<< "BmSym" << " mflop/sec: " << (flops * UNROLL  / (1000000 * time)) <<endl;
#endif

#ifdef BWTEST
	transform(y_bmsym.getarr(), y_bmsym.getarr() + m, y_bmsym.getarr(), bind2nd(divides<double>(), static_cast<double>(UNROLL)));
	cout << "Mega register blocks per second: " << (bmsym.numregb() * UNROLL) / (1000000 * time) << endl;
#endif
	//---------------------------------------//

	y_cscsym+= (*cscsym) * x;
            
	t0 = timer_seconds_since_init();
	for(int i=0; i < REPEAT; ++i)
	{
	    	y_cscsym += (*cscsym) * x;
	}
	t1 = timer_seconds_since_init();
	time = (t1-t0)/REPEAT;
	cout<< "CscSym" << " time: " << time << " seconds" <<endl;
	cout<< "CscSym" << " mflop/sec: " << (flops  / (1000000 * time)) <<endl;

	// Verify with unsymmetric CSC (serial)
	y_csc += (*csc) * x;
            
	t0 = timer_seconds_since_init();
	for(int i=0; i < REPEAT; ++i)
	{
	    	y_csc += (*csc) * x;
	}
	t1 = timer_seconds_since_init();
	time = (t1-t0)/REPEAT;
	cout<< "CSC" << " time: " << time << " seconds" <<endl;
        cout<< "CSC" << " mflop/sec: " << flops / (1000000 * time) <<endl;

	Verify(y_csc, y_cscsym, "CscSym", m);
	Verify(y_csc, y_csbsym, "CsbSym", m);
	Verify(y_csc, y_bmsym, "BmSym", m);

	delete csc;
	delete cscsym;
}

