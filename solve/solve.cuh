#ifndef SOLVE_H
#define SOLVE_H

using namespace std;

__global__ void computeBoundariesLines(double *d_phi, const int nx, const int ny);
__global__ void computeBoundariesColumns(double *d_phi, const int nx, const int ny);
__global__ void copyPhi(double *phi, double *phi_n, const int nx, const int ny);
__global__ void solveAdvectionEquationExplicit(double *phi, double *phi_n, double *u, double *v, const int nx, const int ny, const double dx, const double dy, const double dt);
#endif // SOLVE_H
