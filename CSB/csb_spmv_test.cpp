#define NOMINMAX
#include <iostream> 
#include <algorithm>
#include <numeric>
#include <functional>
#include <fstream>
#include <ctime>
#include <cmath>
#include <string>

#include "timer.gettimeofday.c"
#include "cilk_util.h"
#include "utility.h"

#include "triple.h"
#include "csc.h"
#include "bicsb.h"
#include "bmcsb.h"
#include "spvec.h"
#include "Semirings.h"

using namespace std;

#define INDEXTYPE uint32_t
#define VALUETYPE float

/* Alternative native timer (wall-clock):
 *	timeval tim;		
 *	gettimeofday(&tim, NULL);
 *	double t1=tim.tv_sec+(tim.tv_usec/1000000.0);
 */

int main(int argc, char* argv[])
{
#ifndef CILK_STUB
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
		cout << "Normal usage: ./a.out inputmatrix.mtx sym/nosym binary/text triples/csc" << endl;
		cout << "Assuming matrix.txt is the input, matrix is unsymmetric, and stored in text(ascii) file" << endl;
		inputname = "matrix.txt";
	}
	else if(argc < 3)
	{
		cout << "Normal usage: ./a.out inputmatrix.mtx sym/nosym binary/text triples/csc" << endl;
		cout << "Assuming that the matrix is unsymmetric, and stored in text(ascii) file" << endl;
		inputname =  argv[1];
	}
	else if(argc < 4)
	{
		cout << "Normal usage: ./a.out inputmatrix.mtx sym/nosym binary/text triples/csc" << endl;
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
			forcelogbeta = atoi(argv[5]);
	}

	typedef PTSR<VALUETYPE,VALUETYPE> PTDD;		
	Csc<VALUETYPE, INDEXTYPE> * csc;
	if(binary)
	{
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
		csc= new Csc<VALUETYPE,INDEXTYPE>(triples, nnz, m, n);
		delete [] triples;
	}

	cout << "# workers: "<< gl_nworkers << endl;
	BiCsb<VALUETYPE, INDEXTYPE> bicsb(*csc, gl_nworkers);
	// BiCsb<bool, INDEXTYPE> bin_csb(*csc, gl_nworkers);

#ifndef NOBM
	BmCsb<VALUETYPE, INDEXTYPE, RBDIM> bmcsb(*csc, gl_nworkers);
	ofstream stats("stats.txt");	
	bmcsb.PrintStats(stats);
	Spvec<VALUETYPE, INDEXTYPE> y_bmcsb(m);
	y_bmcsb.fillzero();
#endif
		
	INDEXTYPE flops = 2 * nnz;
	cout << "generating vectors... " << endl;
	Spvec<VALUETYPE, INDEXTYPE> x(n);
	Spvec<VALUETYPE, INDEXTYPE> y_bicsb(m);
	Spvec<VALUETYPE, INDEXTYPE> y_csc(m);
	y_csc.fillzero();
	y_bicsb.fillzero();
	x.fillfota();
	
	cout << "starting SpMV ... " << endl;
	cout << "Row imbalance is: " << RowImbalance(bicsb) << endl;
	timer_init();

	bicsb_gespmv<PTDD>(bicsb, x.getarr(), y_bicsb.getarr());

	double t0 = timer_seconds_since_init();
	for(int i=0; i < REPEAT; ++i)
	{
		bicsb_gespmv<PTDD>(bicsb, x.getarr(), y_bicsb.getarr());
	}
	double t1 = timer_seconds_since_init();

	double time = (t1-t0)/REPEAT;
	cout<< "BiCSB" << " time: " << time << " seconds" <<endl;
	cout<< "BiCSB" << " mflop/sec: " << flops  / (1000000 * time) <<endl;

	/*************************************************************/
#ifndef NOBM
	cout << "starting SpMV with BmCSB ... " << endl;
	cout << "Row imbalance is: " << RowImbalance(bmcsb) << endl;

	prescantime = 0;
	bmcsb_gespmv(bmcsb, x.getarr(), y_bmcsb.getarr());
	prescantime = 0;

	t0 = timer_seconds_since_init();
	for(int i=0; i < REPEAT; ++i)
	{
		bmcsb_gespmv(bmcsb, x.getarr(), y_bmcsb.getarr());
	}
	t1 = timer_seconds_since_init();

	double bmtime = ((t1-t0)/REPEAT) - (prescantime/REPEAT);
	cout<< "BmCSB" << " time: " << bmtime << " seconds" <<endl;
	cout<< "BmCSB" << " mflop/sec: " <<  (flops * UNROLL) / (1000000 * bmtime) <<endl;

#ifdef BWTEST
	transform(y_bmcsb.getarr(), y_bmcsb.getarr() + m, y_bmcsb.getarr(), bind2nd(divides<double>(), static_cast<double>(UNROLL)));
	cout << "Mega register blocks per second: " << (bmcsb.numregb() * UNROLL) / (1000000 * bmtime) << endl;
#endif	// BWTEST
	cout<< "Prescan time: " << (prescantime/REPEAT)  << " seconds" <<endl;
#endif	// NOBM

	// Verify with CSC (serial)
	y_csc += (*csc) * x;
        
	t0 = timer_seconds_since_init();
	for(int i=0; i < REPEAT; ++i)
	{
	    	y_csc += (*csc) * x;
	}
	t1 = timer_seconds_since_init();
	double csctime = (t1-t0)/REPEAT;
	cout<< "CSC" << " time: " << csctime << " seconds" <<endl;
    cout<< "CSC" << " mflop/sec: " << flops / (1000000 * csctime) <<endl;

	Verify(y_csc, y_bicsb, "BiCSB", m);
#ifndef NOBM
	Verify(y_csc, y_bmcsb, "BmCSB", m);
#endif
	delete csc;
}

