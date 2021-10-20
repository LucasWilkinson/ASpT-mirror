CILK = /home/kunal/intel/compilers_and_libraries/linux




INCADD = -I$(CILK)/include -I$(CILK)/examples/include
LIBADD = -L$(CILK)/lib/intel64

GCCOPT = -O2 -fno-rtti -fno-exceptions # -ftree-vectorize
INTELOPT = -O2 -no-ipo -fno-rtti -fno-exceptions -parallel -restrict -std=c++11 -xMIC-AVX512 -no-prec-div #-fno-inline-functions
DEB = -g -DNOBM -O0 -parallel -restrict -std=c++11 

#seqsym: sym_spmv_test.cpp csbsym.cpp csbsym.h utility.h friends.h SSEspmv.o
#	icpc -cilk-serialize $(INCADD) $(INTELOPT) -o seqsym sym_spmv_test.cpp SSEspmv.o

#parsym: sym_spmv_test.cpp csbsym.cpp csbsym.h utility.h friends.h SSEspmv.o
#	icpc $(INCADD) $(DEB) -o parsym sym_spmv_test.cpp SSEspmv.o 

#symanal: sym_spmv_test.cpp csbsym.cpp csbsym.h utility.h friends.h SSEspmv.o
#	icpc -DSTATS $(INCADD) $(INTELOPT) -o symanal sym_spmv_test.cpp SSEspmv.o -lcilkutil

#seqspmv: csb_spmv_test.cpp bicsb.cpp bicsb.h bmcsb.cpp bmcsb.h friends.h utility.h SSEspmv.o
#	icpc -cilk-serialize $(INCADD) $(INTELOPT) -o seqspmv csb_spmv_test.cpp SSEspmv.o

#parspmv: csb_spmv_test.cpp bicsb.cpp bicsb.h bmcsb.cpp bmcsb.h friends.h utility.h SSEspmv.o 
#	icpc $(INCADD) $(INTELOPT) -o parspmv csb_spmv_test.cpp SSEspmv.o

#parspmv_nobm: csb_spmv_test.cpp bicsb.cpp bicsb.h friends.h utility.h
#	icpc $(INCADD) $(INTELOPT) -DNOBM -o parspmv_nobm csb_spmv_test.cpp

#parspmvt: csb_spmvt_test.cpp bicsb.cpp bicsb.h utility.h friends.h
#	icpc $(INCADD) $(INTELOPT) -o parspmvt csb_spmvt_test.cpp

#both_d:	both_test.cpp bicsb.cpp bicsb.h utility.h friends.h
#	icpc $(INCADD) $(INTELOPT) -o both_d both_test.cpp

#both_s:	both_test.cpp bicsb.cpp bicsb.h utility.h friends.h
#	icpc $(INCADD) $(INTELOPT) -DSINGLEPRECISION -o both_s both_test.cpp

spmm_dall:	spmm_testd.cpp bicsb.cpp bicsb.h utility.h friends.h
	for number in 32 128 512; do \
		echo "icpc $(INCADD) $(INTELOPT) -DRHSDIM=$$number -o spmm_d$$number spmm_testd.cpp"; \
		icpc $(INCADD) $(INTELOPT) -DRHSDIM=$$number -o spmm_d$$number spmm_testd.cpp; \
	done;

spmm_a:	spmm_test.cpp bicsb.cpp bicsb.h utility.h friends.h
	icpc $(INCADD) $(INTELOPT) -DSINGLEPRECISION -S -fcode-asm -vec_report6 spmm_test.cpp

spmm_sall:	spmm_test.cpp bicsb.cpp bicsb.h utility.h friends.h
	for number in 32 128 512; do \
		echo "icpc $(INCADD) $(INTELOPT) -DSINGLEPRECISION -DRHSDIM=$$number -o spmm_s$$number spmm_test.cpp"; \
		icpc $(INCADD) $(INTELOPT) -DSINGLEPRECISION -DRHSDIM=$$number -o spmm_s$$number spmm_test.cpp; \
	done;

SSEspmv.o: SSEspmv.cpp
	g++ -DAMD $(GCCOPT) -march=amdfam10 -c SSEspmv.cpp	

clean:	
	rm -f seqspmv
	rm -f seqsym
	rm -f parspmv
	rm -f parsym 
	rm -f parspmvt
	rm -f parspmv_nobm
	for number in 8 16 24 32 40 48 56 64; do \
		rm -f spmm_s$$number;\
		rm -f spmm_d$$number;\
	done;
	rm -f *.o
