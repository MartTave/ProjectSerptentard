#include <stdio.h>

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

__global__ void copyPhi(double *phi, double *phi_n, const int nx, const int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny)
    {
        return;
    }

    phi_n[i * ny + j] = phi[i * ny + j];
}

__global__ void solveAdvectionEquationExplicit(double *phi, double *phi_n, double *u, double *v, const int nx, const int ny, const double dx, const double dy, const double dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny)
    {
        return;
    }
    int l = i * ny + j;
    if (u[l] > 0.0)
    {
        phi[l] -= dt * (u[l] * (phi_n[(i + 1) * ny + j] - phi_n[l]) / dx);
    }
    else
    {
        phi[l] -= dt * (u[l] * (phi_n[i * ny + j] - phi_n[(i - 1) * ny + j]) / dx);
    }
    if (v[l] < 0.0)
    {
        phi[l] -= dt * (v[l] * (phi_n[l + 1] - phi_n[l]) / dy);
    }
    else
    {
        phi[l] -= dt * (v[l] * (phi_n[l] - phi_n[l - 1]) / dy);
    }
}