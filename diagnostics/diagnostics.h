#ifndef DIAGNOSTICS_H
#define DIAGNOSTICS_H

using namespace std;

void computeInterfaceLength(double** phi, const int nx, const int ny, const double dx, const double dy);
void computeInterfaceCurvature(double** phi,double** curvature, const int nx, const int ny, const double dx, const double dy);

#endif // DIAGNOSTICS_H
