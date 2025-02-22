#ifndef __FILEH__
#define __FILEH__

#include <sys/time.h>

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

#ifdef __cplusplus
extern "C" {
#endif
void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);
// void print_matrix(double* A, int M, int N);
#ifdef __cplusplus
}
#endif

#define FATAL(msg, ...) \
    do {\
      fprintf(stderr, "[%s:%d] "msg"\n", __FILE__, __LINE__, ##__VA_ARGS__); \
      exit(-1);								\
    } while(0)

#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

#endif
