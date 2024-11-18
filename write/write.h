#ifndef WRITE_H
#define WRITE_H

#include <string>
#include <vector>

using namespace std;

void writeDataVTK(const string filename, double *phi, double *curvature, double *u, double *v, const int nx, const int ny, const double dx, const double dy, const int step);
string getString(double *data, long size)
#endif // WRITE_H
