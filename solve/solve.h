#ifndef SOLVE_H
#define SOLVE_H

using namespace std;

void computeBoundaries(double** phi, const int nx, const int ny);
void solveAdvectionEquationExplicit(
    double** phi, double** u, double** v, const int nx, const int ny, const double dx, const double dy, const double dt);

#endif // SOLVE_H
