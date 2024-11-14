#ifndef INIT_H
#define INIT_H

using namespace std;

__global__ void InitializationKernel(double* phi, double* curvature, double* u, double* v, const int nx, const int ny, const double dx, const double dy);

#endif // INIT_H