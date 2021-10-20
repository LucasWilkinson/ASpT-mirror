// -*- C++ -*-

#ifndef CILK_UTIL_H_INCLUDED
#define CILK_UTIL_H_INCLUDED

#ifdef _WIN32
#include <Windows.h>
#else
#include <stdlib.h>
#include <sys/time.h>
#endif

/* cilk_get_time
   get the time in milliseconds.  This means different things in Windows vs.
   Unix.  In Windows, it's a call to GetTickCount() which is the uptime of
   the system.  In Unix, it is implemented with gettimeofday() and sets the
   counter to zero the first time cilk_get_time is called.

   returns: the number of milliseconds since the start time.
 */
extern "C++"
inline
int cilk_get_time () {
#ifdef _WIN32
    // Windows implementation.
    return (int) GetTickCount();
#else
    static timeval *start = NULL;
    struct timezone tzp = { 0, 0 };
    if (NULL == start) {
        // define the current time as 0.
        start = (timeval*) malloc(sizeof(timeval));
        gettimeofday(start, &tzp);
        return 0;
    } else {
        // subtract the start time from the current time.
        timeval end;
        long ms = 0;
        gettimeofday(&end, &tzp);
        ms = (end.tv_sec - start->tv_sec) * 1000;
        ms += ((int)end.tv_usec - (int)start->tv_usec) / 1000;
        return (int) ms;
    }
#endif
}
#endif // CILK_UTIL_H_INCLUDED
