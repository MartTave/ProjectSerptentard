#include "solve.h"

using namespace std;

// Compute the boundaries of the domain for the phi field
void computeBoundaries(double **phi, const int nx, const int ny)
{

    // Upper and Lower boundaries (extrapolation)
    for (int i = 0; i < nx; i++)
    {
        phi[i][ny - 1] = 2.0 * phi[i][ny - 2] - phi[i][ny - 3];
        phi[i][0] = 2.0 * phi[i][1] - phi[i][2];
    }

    // Left and Right boundaries (extrapolation)
    for (int j = 0; j < ny; j++)
    {
        phi[0][j] = 2.0 * phi[1][j] - phi[2][j];
        phi[nx - 1][j] = 2.0 * phi[nx - 2][j] - phi[nx - 3][j];
    }
}

// Solving advection equation on the inside domain ([1;nx-2] x [1;ny-2])
// Equation solved: d phi / dt + u d phi / dx + v d phi / dy = 0
// Using the euler explicit numerical scheme => phi = phi_n - (u d phi / dx + v d phi / dy)
// A first order upwind scheme is used to stabilize the solver (https://en.wikipedia.org/wiki/Upwind_scheme)
void solveAdvectionEquationExplicit(
    double **phi, double **u, double **v, const int nx, const int ny, const double dx, const double dy, const double dt)
{

    // Copy phi to phi_n
    double **phi_n = new double *[nx]; // Previous phi field
    for (int i = 0; i < nx; i++)
    {
        phi_n[i] = new double[ny];
        for (int j = 0; j < ny; j++)
        {
            phi_n[i][j] = phi[i][j];
        }
    }

    // Compute the advection equation
    for (int i = 1; i < nx - 1; i++)
    {
        for (int j = 1; j < ny - 1; j++)
        {

            phi[i][j] = phi_n[i][j];

            // First order upwind numerical scheme to ensure numerical stability
            if (u[i][j] < 0.0)
            { // x-direction
                phi[i][j] -= dt * (u[i][j] * (phi_n[i + 1][j] - phi_n[i][j]) / dx);
            }
            else
            {
                phi[i][j] -= dt * (u[i][j] * (phi_n[i][j] - phi_n[i - 1][j]) / dx);
            }

            if (v[i][j] < 0.0)
            { // y-direction
                phi[i][j] -= dt * (v[i][j] * (phi_n[i][j + 1] - phi_n[i][j]) / dy);
            }
            else
            {
                phi[i][j] -= dt * (v[i][j] * (phi_n[i][j] - phi_n[i][j - 1]) / dy);
            }
        }
    }

    // Refresh the boundaries values
    computeBoundaries(phi, nx, ny);

    // Deallocate memory
    for (int i = 0; i < nx; ++i)
    {
        delete[] phi_n[i];
    }
    delete[] phi_n;
}