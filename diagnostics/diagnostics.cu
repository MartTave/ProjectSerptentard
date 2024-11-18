#include <math.h>
#include <iostream>

#include "diagnostics.cuh"

using namespace std;

__global__ void computeInterfaceLengthKernel(double* phi, double* lengths, const int nx, const int ny, const double dx, const double dy) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) {
        return;
    }

    // Fixed parameter for the dirac function
    double epsilon = 0.001;
    
    // Interface length
    double length = 0.0;

    // Compute gradient of phi : grad(phi)
    double phi_x = (phi[(i+1) * ny + j]-phi[(i-1) * ny + j])/2.0/dx;
    double phi_y = (phi[i * ny + j+1]-phi[i * ny + j-1])/2.0/dy;

    // Compute the norm of gradient: norm(grad(phi))
    double normGrad = sqrt(phi_x*phi_x + phi_y*phi_y);

    // Compute the dirac function approximation
    double delta = (1.0 / sqrt(2.0 * M_PI * epsilon)) * exp( - (phi[i * ny + j] * phi[i * ny + j]) / (2.0 * epsilon) ) ;

    // L = delta * norm(grad(phi)) * dx * dy
    length = delta * normGrad * dx * dy;

    lengths[i * ny + j] = length;
}