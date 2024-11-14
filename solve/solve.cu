#include "solve.cuh"

__global__ void computeBoundariesLines(double *d_phi, const int nx, const int ny)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    // First line
    if (n < nx)
    {
        d_phi[n] = 2.0 * d_phi[n + 1 * nx] - d_phi[n + 2 * nx];
    }
    // Last line
    int lastLineIndex = n + ((ny - 1) * nx);
    if (lastLineIndex < nx * ny)
    {
        d_phi[lastLineIndex] = 2.0 * d_phi[n + (ny - 2) * nx] - d_phi[n + (ny - 3) * nx];
    }
}

__global__ void computeBoundariesColumns(double *d_phi, const int nx, const int ny)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    // First col
    if (n < ny)
    {
        d_phi[n * nx] = 2.0 * d_phi[n * nx + 1] - d_phi[n * nx + 2];
    }
    // Last col
    int lastColIndex = n * nx + nx - 1;
    if (lastColIndex < nx * ny)
    {
        d_phi[lastColIndex] = 2.0 * d_phi[n * nx + nx - 2] - d_phi[n * nx + nx - 3];
    }
}