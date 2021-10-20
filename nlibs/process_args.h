#ifndef PROCESS_ARGS_H
#define PROCESS_ARGS_H

#include <fcntl.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "qrmcl.h"

enum SharedOption {CachePreferNone, CachePreferShared, CachePreferL1};
static char *sharedOptionsStr[] = {"CachePreferNone", "CachePreferShared", "CachePreferL1"};
static std::pair<string, SharedOption> smap_data[] = {
    std::make_pair("None", CachePreferNone),
    std::make_pair("Shared", CachePreferShared),
    std::make_pair("L1", CachePreferL1),
    std::make_pair("none", CachePreferNone),
    std::make_pair("shared", CachePreferShared),
    std::make_pair("l1", CachePreferL1),
};
static std::map<std::string, SharedOption> sharedMap(smap_data,
        smap_data + sizeof smap_data / sizeof smap_data[0]);

struct Options {
  bool calcChange = false;
  int maxIters = 5;
  char inputFileName[200];
  char inputFileNameB[200];
  bool stats = false;
  int stride = 512;
  int ptile = 2;
  RunOptions rmclOption;
  SharedOption sharedOption;
  int br = 2, bc = 8;
  int nbins = 13;
  int hashing = 0;
  int refinedESC = 0;
  int memory_managed = 0;
  int diffMatrices = 0;
  int Bcolpartition = 0;
  int scatterVector = 0;
  int twophase = 0;
  int chunkSizeFactor = 1;
  int chunkSize = 1024;
};

extern Options options;

int process_args(int argc, char **argv);
void print_args();
#endif
