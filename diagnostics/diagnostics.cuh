#ifndef DIAGNOSTICS_H
#define DIAGNOSTICS_H

using namespace std;

__global__ void computeInterfaceLengthKernel(double* phi, const int nx, const int ny, const double dx, const double dy);

#endif // DIAGNOSTICS_H
