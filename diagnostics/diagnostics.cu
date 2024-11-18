#include <math.h>
#include <iostream>

#include "diagnostics.cuh"

using namespace std;

__global__ void computeInterfaceLengthKernel(double *phi, double *lengths, const int nx, const int ny, const double dx, const double dy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny)
    {
        return;
    }

    // Fixed parameter for the dirac function
    double epsilon = 0.001;

    // Compute gradient of phi : grad(phi)
    double phi_x = (phi[(i + 1) * ny + j] - phi[(i - 1) * ny + j]) / 2.0 / dx;
    double phi_y = (phi[i * ny + j + 1] - phi[i * ny + j - 1]) / 2.0 / dy;

    // Compute the norm of gradient: norm(grad(phi))
    double normGrad = sqrt(phi_x * phi_x + phi_y * phi_y);

    // Compute the dirac function approximation
    double delta = (1.0 / sqrt(2.0 * M_PI * epsilon)) * exp(-(phi[i * ny + j] * phi[i * ny + j]) / (2.0 * epsilon));

    lengths[i * ny + j] = delta * normGrad * dx * dy;
}

__global__ void computeInterfaceCurvatureKernel(double *phi, double *curvature, const int nx, const int ny, const double dx, const double dy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1)
    {
        return;
    }

    if (abs(phi[i + j * nx]) < 3.0 * dx)
    { // Compute the curvature only near the interface

        // first derivative
        double phi_x = (phi[i + 1 + j * nx] - phi[i - 1 + j * nx]) / 2.0 / dx;
        double phi_y = (phi[i + (j + 1) * nx] - phi[i + (j - 1) * nx]) / 2.0 / dy;

        // second derivative
        double phi_xx = (phi[i + 1 + j * nx] - 2.0 * phi[i + j * nx] + phi[i - 1 + j * nx]) / dx / dx;
        double phi_yy = (phi[i + (j + 1) * nx] - 2.0 * phi[i + j * nx] + phi[i + (j - 1) * nx]) / dy / dy;
        double phi_xy = (phi[i + 1 + (j + 1) * nx] - phi[i + 1 + (j - 1) * nx] - phi[i - 1 + (j + 1) * nx] + phi[i - 1 + (j - 1) * nx]) / dx / dy / 4.0;

        // compute curvature
        curvature[i + j * nx] = (phi_xx * phi_y * phi_y - 2.0 * phi_x * phi_y * phi_xy + phi_yy * phi_x * phi_x) / pow(phi_x * phi_x + phi_y * phi_y, 1.5);
    }
    else
    { // Default value if the cell is not closed to the interface
        curvature[i + j * nx] = 0.0;
    }
}