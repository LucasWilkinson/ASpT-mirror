#ifndef QRMCL_H_
#define QRMCL_H_
#include "CSR.h"
#include "COO.h"
#include <map>
#include <string>

enum RunOptions {SEQ, OMP, GPU, CILK, SOMP, MKL, SFOMP, HYB};
static char *runOptionsStr[] = {"SEQ", "OMP", "GPU", "CILK", "SOMP", "MKL", "SFOMP", "HYB"};
static std::pair<string, RunOptions> map_data[] = {
    std::make_pair("SEQ", SEQ),
    std::make_pair("OMP", OMP),
    std::make_pair("GPU", GPU),
    std::make_pair("CILK", CILK),
    std::make_pair("SOMP", SOMP),
    std::make_pair("MKL", MKL),
    std::make_pair("SFOMP", SFOMP),
    std::make_pair("HYB", HYB),
};
static std::map<std::string, RunOptions> runMap(map_data,
        map_data + sizeof map_data / sizeof map_data[0]);

CSR rmclInit(COO &cooAt);
CSR RMCL(const char iname[], int maxIters, RunOptions runOptions);
#endif
