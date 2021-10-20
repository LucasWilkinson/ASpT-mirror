#ifndef _MYTIMER_
#define _MYTIMER_


#include <time.h>

struct timespec timer_ApplicationStartTime;
int timer_initialized = 0;

void timer_init(){
  if(timer_initialized){fprintf(stderr,"timer_init() must be called once and only once\n");exit(0);}
  timer_initialized = 1;
  clock_gettime(CLOCK_REALTIME,&timer_ApplicationStartTime);
}

double timer_seconds_since_init(){
  if(!timer_initialized){fprintf(stderr,"timer_init() must be called first\n");exit(0);}
  struct timespec timer_CurrentTime;
  clock_gettime(CLOCK_REALTIME,&timer_CurrentTime);
  double rv = 1.0*(timer_CurrentTime.tv_sec-timer_ApplicationStartTime.tv_sec)+1e-9*(timer_CurrentTime.tv_nsec-timer_ApplicationStartTime.tv_nsec);
  return(rv);
}

#endif
