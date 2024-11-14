// Libraries
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <cuda.h>

// == User lib ==
#include "diagnostics/diagnostics.h"
#include "initialization/init.h"
#include "solve/solve.cuh"
#include "write/write.h"

#include "common_includes.c"

// Namespace
using namespace std;

// Advection Solver
int main(int argc, char *argv[])
{

    // Data Initialization
    // == Spatial ==
    int scale = 1;
    if (argc > 1)
    {
        scale = stoi(argv[1]);
    }

    int nx = 100 * scale;
    int ny = 100 * scale; // Number of cells in each direction
    double Lx = 1.0;
    double Ly = 1.0; // Square domain [m]
    double dx = Lx / (nx - 1);
    double dy = Ly / (ny - 1); // Spatial step [m]

    // == Temporal ==
    double tFinal = 2.0;           // Final time [s]
    double dt = 0.005 / scale;     // Temporal step [s]
    int nSteps = int(tFinal / dt); // Number of steps to perform
    double time = 0.0;             // Actual Simulation time [s]

    // == Numerical ==
    int outputFrequency = nSteps / 40;

    double *h_phi;
    double *h_curvature;
    double *h_u;
    double *h_v;

    double *d_phi;
    double *d_phi_n;
    double *d_curvature;
    double *d_u;
    double *d_v;

    CHECK_ERROR(cudaMalloc((void **)&d_phi, nx * ny));
    CHECK_ERROR(cudaMalloc((void **)&d_phi_n, nx * ny));
    CHECK_ERROR(cudaMalloc((void **)&d_curvature, nx * ny));
    CHECK_ERROR(cudaMalloc((void **)&d_u, nx * ny));
    CHECK_ERROR(cudaMalloc((void **)&d_v, nx * ny));

    Initialization(d_phi, d_curvature, d_u, d_v, nx, ny, dx, dy); // Initialize the distance function field
    computeBoundariesLines<<<>>>(d_phi, nx, ny);
    computeBoundariesColumns<<<>>>(d_phi, nx, ny);
    cudaDeviceSyncronize();

    cudaMemcpy(d_phi, h_phi, nx * ny, cudaMemcpyDeviceToHost);

    printBeginAndEnd(5, h_phi, nx * ny);

    return 0;

    // == Output ==
    stringstream ss;
    ss << scale;
    string scaleStr = ss.str();

    string outputName = "output/levelSet_scale" + scaleStr + "_";
    int count = 0; // Number of VTK file already written

    // == First output ==
    // Write data in VTK format
    mkdir("output", 0777); // Create output folder

    // TODO: Memcopy from device to host
    writeDataVTK(outputName, phi, curvature, u, v, nx, ny, dx, dy, count++);

    // Loop over time
    for (int step = 1; step <= nSteps; step++)
    {

        time += dt; // Simulation time increases
        cout << "\nStarting iteration step " << step << "/" << nSteps << "\tTime " << time << "s\n";

        // Solve the advection equation
        solveAdvectionEquationExplicit(phi, u, v, nx, ny, dx, dy, dt);

        cudaDeviceSyncronize();

        // Diagnostics: interface perimeter
        computeInterfaceLength(phi, nx, ny, dx, dy);

        // Diagnostics: interface curvature
        computeInterfaceCurvature(phi, curvature, nx, ny, dx, dy);

        cudaDeviceSyncronize();

        // TODO: Memcopy from device to host (This time, no need to copy u and v)

        // Write data to output file
        if (step % outputFrequency == 0)
        {
            writeDataVTK(outputName, phi, curvature, u, v, nx, ny, dx, dy, count++);
        }
    }

    // Free memory
    delete[] h_phi, h_curvature, h_u, h_v;

    CHECK_ERROR(cudaFree((void **)&d_phi));
    CHECK_ERROR(cudaFree((void **)&d_phi_n));
    CHECK_ERROR(cudaFree((void **)&d_curvature));
    CHECK_ERROR(cudaFree((void **)&d_u));
    CHECK_ERROR(cudaFree((void **)&d_v));

    return 0;
}
