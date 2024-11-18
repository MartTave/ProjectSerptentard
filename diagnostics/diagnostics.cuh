#ifndef DIAGNOSTICS_H
#define DIAGNOSTICS_H

using namespace std;

__global__ void computeInterfaceLengthKernel(double *phi, double *lengths, const int nx, const int ny, const double dx, const double dy);

__global__ void computeInterfaceCurvatureKernel(double *phi, double *curvature, const int nx, const int ny, const double dx, const double dy);
#endif // DIAGNOSTICS_H
