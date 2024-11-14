#include <math.h>

#include "../diagnostics/diagnostics.h"
#include "init.h"

using namespace std;

// Initialization of the distance function inside the domain
void Initialization(double **phi, double **curvature, double **u, double **v, const int nx, const int ny, const double dx, const double dy)
{

    // == Circle parameters ==
    double xcenter = 0.5;  // Circle position x
    double ycenter = 0.75; // Circle position y
    double radius = 0.15;  // Circle radius

    for (int i = 0; i < nx; i++)
    {
        double x = i * dx - xcenter;

        for (int j = 0; j < ny; j++)
        {
            double y = j * dy - ycenter;

            // Compute the signed distance to the interface
            double distance = sqrt(x * x + y * y) - radius;

            phi[i][j] = distance;

            // Compute the velocity based on x and y
            u[i][j] = sin(2.0 * M_PI * j * dy) * sin(M_PI * i * dx) * sin(M_PI * i * dx);
            v[i][j] = -sin(2.0 * M_PI * i * dx) * sin(M_PI * j * dy) * sin(M_PI * j * dy);
        }
    }

    // Compute the interface curvature
    computeInterfaceCurvature(phi, curvature, nx, ny, dx, dy);
}
