#ifndef SOLVE_H
#define SOLVE_H

using namespace std;

__global__ void computeBoundariesLines(double *d_phi, const int nx, const int ny);
__global__ void computeBoundariesColumns(double *d_phi, const int nx, const int ny);
#endif // SOLVE_H
