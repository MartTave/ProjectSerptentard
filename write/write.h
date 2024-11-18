#ifndef WRITE_H
#define WRITE_H

#include <string>
#include <vector>

using namespace std;

void writeDataVTK(const string filename, string phi_part, string curvature_part, string u_part, string v_part, const int nx, const int ny, const double dx, const double dy, const int step, const int world_rank);
string getString(double *data, long size);
#endif // WRITE_H
