#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include "timer.gettimeofday.c"
//#include "timer.clock_gettime.c"

int main(){
  int i;
  timer_init();
  for(i=0;i<20;i++){
    double t0 = timer_seconds_since_init();
    sleep(1);
    double t1 = timer_seconds_since_init();
    printf("%12.9f, %12.9f\n",t0,t1-t0);
  }
}
