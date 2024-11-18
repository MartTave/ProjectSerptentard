// Libraries
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>

#include <cuda.h>

// == User lib ==
#include "diagnostics/diagnostics.cuh"
#include "initialization/init.cuh"
#include "solve/solve.cuh"
#include "write/write.h"

#include "common_includes.cu"

// Namespace
using namespace std;

// Advection Solver
int main(int argc, char *argv[])
{

    // Data Initialization
    // == Spatial ==
    int scale = 10;
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

    double *h_phi = new double[nx * ny];
    double *h_curvature = new double[nx * ny];
    double *h_u = new double[nx * ny];
    double *h_v = new double[nx * ny];
    double *h_lengths = new double[nx * ny];

    double *d_phi;
    double *d_phi_n;
    double *d_curvature;
    double *d_lengths;
    double *d_u;
    double *d_v;

    long size = nx * ny * sizeof(double);

    CHECK_ERROR(cudaMalloc((void **)&d_phi, size));
    CHECK_ERROR(cudaMalloc((void **)&d_lengths, size));
    CHECK_ERROR(cudaMalloc((void **)&d_phi_n, size));
    CHECK_ERROR(cudaMalloc((void **)&d_curvature, size));
    CHECK_ERROR(cudaMalloc((void **)&d_u, size));
    CHECK_ERROR(cudaMalloc((void **)&d_v, size));

    int windowSize = 25;
    int gridWidth = (nx + windowSize - 1) / windowSize;
    int gridHeight = (ny + windowSize - 1) / windowSize;

    dim3 dimGrid(gridWidth, gridHeight);
    dim3 dimBlock(windowSize, windowSize);
    InitializationKernel<<<dimGrid, dimBlock>>>(d_phi, d_curvature, d_u, d_v, nx, ny, dx, dy);
    cudaDeviceSynchronize();
    // TODO: computeInterfaceSignature ?
    computeBoundariesLines<<<1, nx>>>(d_phi, nx, ny);
    computeBoundariesColumns<<<1, ny>>>(d_phi, nx, ny);
    cudaDeviceSynchronize();
    CHECK_ERROR(cudaMemcpy(h_phi, d_phi, size, cudaMemcpyDeviceToHost));

    printBeginAndEnd(150, h_phi, nx * ny);

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
    writeDataVTK(outputName, h_phi, h_curvature, h_u, h_v, nx, ny, dx, dy, count++);

    // Loop over time
    for (int step = 1; step <= nSteps; step++)
    {

        time += dt; // Simulation time increases

        // Solve the advection equation
        copyPhi<<<dimGrid, dimBlock>>>(d_phi, d_phi_n, nx, ny);
        solveAdvectionEquationExplicit<<<dimGrid, dimBlock>>>(d_phi, d_phi_n, d_u, d_v, nx, ny, dx, dy, dt);

        cudaDeviceSynchronize();

        computeBoundariesLines<<<1, nx>>>(d_phi, nx, ny);
        computeBoundariesColumns<<<1, ny>>>(d_phi, nx, ny);

        cudaDeviceSynchronize();

        // Diagnostics: interface perimeter
        computeInterfaceLengthKernel<<<dimGrid, dimBlock>>>(d_phi, d_lengths, nx, ny, dx, dy);

        // Diagnostics: interface curvature
        computeInterfaceCurvatureKernel<<<dimGrid, dimBlock>>>(d_phi, d_curvature, nx, ny, dx, dy);

        // cudaDeviceSynchronize();

        CHECK_ERROR(cudaMemcpy(h_phi, d_phi, size, cudaMemcpyDeviceToHost));
        CHECK_ERROR(cudaMemcpy(h_lengths, d_lengths, size, cudaMemcpyDeviceToHost));
        CHECK_ERROR(cudaMemcpy(h_curvature, d_curvature, size, cudaMemcpyDeviceToHost));

        double max = 0;
        double total_length = 0;

        for (int i = 0; i < nx * ny; i++)
        {
            if (abs(h_curvature[i]) > max)
            {
                max = abs(h_curvature[i]);
            }
            total_length += h_lengths[i];
        }
        // Write data to output file
        if (step % outputFrequency == 0)
        {
            cout << "Step: " << step << "\n\n";
            writeDataVTK(outputName, h_phi, h_curvature, h_u, h_v, nx, ny, dx, dy, count++);
        }
    }

    // Free memory
    delete[] h_phi, h_curvature, h_u, h_v;

    CHECK_ERROR(cudaFree((void **)d_phi));
    CHECK_ERROR(cudaFree((void **)d_phi_n));
    CHECK_ERROR(cudaFree((void **)d_curvature));
    CHECK_ERROR(cudaFree((void **)d_u));
    CHECK_ERROR(cudaFree((void **)d_v));

    return 0;
}
