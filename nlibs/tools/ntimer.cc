#include "ntimer.h"

double time_in_mill_now() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  double time_in_mill =
    (tv.tv_sec) * 1000.0 + (tv.tv_usec) / 1000.0;
  return time_in_mill;
}
