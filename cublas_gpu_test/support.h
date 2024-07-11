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
void verify_vec(float *A, float *B, float *C, int n);
void verify_mat_mul(float *A, float *B, float *C, int M, int K, int N);
void verify_mat_mul_column_major(float *A, float *B, float *C, int M, int K, int N);
void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);
void print_matrix(float *A, int M, int N);
void cpu_mat_mul(float *A, float *B, float *C, int M, int K, int N);
void verify_dmat_mul(double *A, double *B, double *C, int M, int K, int N);
void cpu_dmat_mul(double *A, double *B, double *C, int M, int K, int N);
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
