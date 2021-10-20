#ifndef QMALLOC_H_
#define QMALLOC_H_

#include<stdlib.h>
#include<stdio.h>
#include<errno.h>

inline void *qmalloc(size_t size,
      const char *func, const int line) {
  void *p = malloc(size);
  if (p == NULL) {
    fprintf(stderr, "%s() at line %d failed: malloc(): %s\n",
        func, line, strerror(errno));
    exit(EXIT_FAILURE);
        }
  return p;
}

inline void* qcalloc (size_t num, size_t size,
      const char *func, const int line) {
  void *p = calloc(num, size);
  if (p == NULL) {
    fprintf(stderr, "%s() at line %d failed: qmalloc(): %s\n",
        func, line, strerror(errno));
    exit(EXIT_FAILURE);
        }
  return p;
}
#endif
