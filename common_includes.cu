/**
 * CUDA related includes
 */
#include <stdio.h>

#define CHECK_ERROR(call)                                                                 \
    {                                                                                     \
        cudaError_t err = call;                                                           \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(err);                                                                    \
        }                                                                                 \
    }

/**
 * Performance monitoring
 */
#include <sys/time.h>
long long timeInMilliseconds(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return (((long long)tv.tv_sec) * 1000) + (tv.tv_usec / 1000);
}

// Macro to measure time taken by a call
#define MEASURE_TIME(call)                              \
    {                                                   \
        long long start = timeInMilliseconds();         \
        call;                                           \
        long long end = timeInMilliseconds();           \
        long long time_taken = end - start;             \
        printf("time taken %lld [msec]\n", time_taken); \
    }

/**
 * Debug helpers
 */
void printBeginAndEnd(int nValues, float *values, int totalSize)
{
    for (int i = 0; i < nValues; i++)
        printf("idx[%i] -> %f\n", i, values[i]);

    printf("...\n");

    for (int i = 0; i < nValues; i++)
    {
        int idx = totalSize - nValues + i;
        printf("idx[%i] -> %f\n", idx, values[idx]);
    }
}


void convertIdx(int gridDimX,int blockDimX,  int blockDimY, int blockIdxX, int blockIdxY, int threadIdxX, int threadIdxY, long *pos)
{
    long UniqueBlockIndex = blockIdxY * gridDimX + blockIdxX;
    long UniqueThreadIndex = UniqueBlockIndex * blockDimY * blockDimX + threadIdxY *
    blockDimX + threadIdxX;
    *pos = UniqueThreadIndex;
}

void get2DPos(int nx, int ny, int idx, int *i, int *j)
{
    *j=(idx+1)%nx;
    *i=idx+1-*j*nx;
}