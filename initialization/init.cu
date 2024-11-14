#include <math.h>

#include "../diagnostics/diagnostics.h"
#include "init.cuh"

using namespace std;


__global__ void InitializationKernel(double *phi, double *curvature, double *u, double *v, const int nx, const int ny, const double dx, const double dy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) {
        return;
    }

    // == Circle parameters ==
    double xcenter = 0.5;  // Circle position x
    double ycenter = 0.75; // Circle position y
    double radius = 0.15;  // Circle radius

    double x = i * dx - xcenter;
    double y = j * dy - ycenter;

    // Compute the signed distance to the interface
    double distance = sqrt(x * x + y * y) - radius;

    if(i != 0 || j != 0 || i != nx - 1 || j != ny - 1) {
        phi[i * ny + j] = distance;
    }

    // Compute the velocity based on x and y
    u[i * ny + j] = sin(2.0 * M_PI * j * dy) * sin(M_PI * i * dx) * sin(M_PI * i * dx);
    v[i * ny + j] = -sin(2.0 * M_PI * i * dx) * sin(M_PI * j * dy) * sin(M_PI * j * dy);
}