include ../Makefile.in

INCLUDES = -I. 

CFLAGS = $(COPTIONS) $(OPTFLAGS) $(CINCLUDES)
CUFLAGS = $(CUOPTFLAGS) $(CUINCLUDES) -I.
LD = $(CC) -L. 

#SOURCES=$(wildcard *.cc)
SOURCES=$(shell find . -path ./gpus -prune -o -path ./cilks -prune -o -path ./mkls -prune -o -name '*.cc' -print) 

ifeq ($(hasGPU), 1)
SOURCES+= $(shell find gpus -name "*.cc" -o -name "*.cu")
endif

ifeq ($(hasCILK), 1)
SOURCES+= $(shell find cilks -name "*.cc")
CFLAGS+= $(MKL_FLAGS)
endif

ifeq ($(hasMKL), 1)
SOURCES+= $(shell find mkls -name "*.cc")
endif

OBJS=$(shell echo $(SOURCES) | sed "s/\.cc/\.o/g" | sed "s/\.cu/\.o/g")
HEADERS=$(shell find . -name "*.h")
%.o: %.cc $(HEADERS) ../Makefile.in
	$(CC) $(CFLAGS) $< -c -o $@

%.o: %.cu ../Makefile.in $(shell find . -name "*.cuh")
	$(NVCC) $(CUFLAGS) $< -c -o $@

$(LIBNAME): $(OBJS) $(HEADERS)
	$(AR) $@ $(OBJS)
	$(RANLIB) $@

source:
	echo $(SOURCES)

clean:
	rm -f *.o gpus/*.o tools/*.o cilks/*.o mkls/*.o
	rm -f *.a

