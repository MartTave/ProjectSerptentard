// Libraries
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <mpi.h>
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
    MPI_Status status;

    int world_size, world_rank;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Variables declaration
    int nx, ny, nSteps, scale, outputFrequency, gridWidth, gridHeight, windowSize;
    scale = 10;
    if (argc > 1)
    {
        scale = stoi(argv[1]);
    }
    nx = 100 * scale;
    ny = 100 * scale; // Number of cells in each direction

    double Lx, Ly, dx, dy, tFinal, dt, time;

    long arrayLength, arraySplittedSize;
    arrayLength = nx * ny;
    stringstream ss;

    int count = 0; // Number of VTK file already written
    string scaleStr = ss.str();
    string outputName = "output/levelSet_scale" + scaleStr + "_";

    dim3 dimGrid, dimBlock;

    double *h_phi = new double[arrayLength];
    double *h_curvature = new double[arrayLength + (arrayLength % world_size)];
    double *h_u = new double[arrayLength + (arrayLength % world_size)];
    double *h_v = new double[arrayLength + (arrayLength % world_size)];
    double *h_lengths = new double[arrayLength + (arrayLength % world_size)];
    double *d_phi;
    double *d_phi_n;
    double *d_curvature;
    double *d_lengths;
    double *d_u;
    double *d_v;
    long size = arrayLength * sizeof(double);

    if (world_rank == 0)
    {

        Lx = 1.0;
        Ly = 1.0; // Square domain [m]
        dx = Lx / (nx - 1);
        dy = Ly / (ny - 1); // Spatial step [m]

        // == Temporal ==
        tFinal = 4.0;              // Final time [s]
        dt = 0.005 / scale;        // Temporal step [s]
        nSteps = int(tFinal / dt); // Number of steps to perform
        time = 0.0;                // Actual Simulation time [s]

        // == Numerical ==
        outputFrequency = nSteps / 40;

        arraySplittedSize = (arrayLength + (arrayLength % world_size)) / world_size;

        for (int i = arrayLength; i < arrayLength + (arrayLength % world_size); i++)
        {
            h_curvature[i] = 0;
            h_u[i] = 0;
            h_v[i] = 0;
            h_lengths[i] = 0;
        }
        size = arrayLength * sizeof(double);

        CHECK_ERROR(cudaMalloc((void **)&d_phi, size));
        CHECK_ERROR(cudaMalloc((void **)&d_lengths, size));
        CHECK_ERROR(cudaMalloc((void **)&d_phi_n, size));
        CHECK_ERROR(cudaMalloc((void **)&d_curvature, size));
        CHECK_ERROR(cudaMalloc((void **)&d_u, size));
        CHECK_ERROR(cudaMalloc((void **)&d_v, size));

        windowSize = 25;
        gridWidth = (nx + windowSize - 1) / windowSize;
        gridHeight = (ny + windowSize - 1) / windowSize;
        dimGrid = dim3(gridWidth, gridHeight);
        dimBlock = dim3(windowSize, windowSize);

        InitializationKernel<<<dimGrid, dimBlock>>>(d_phi, d_curvature, d_u, d_v, nx, ny, dx, dy);
        cudaDeviceSynchronize();
        // TODO: computeInterfaceSignature ?
        computeBoundariesLines<<<1, nx>>>(d_phi, nx, ny);
        computeBoundariesColumns<<<1, ny>>>(d_phi, nx, ny);
        cudaDeviceSynchronize();
        CHECK_ERROR(cudaMemcpy(h_phi, d_phi, size, cudaMemcpyDeviceToHost));
        // == Output ==
        ss << scale;

        // == First output ==
        // Write data in VTK format
        mkdir("output", 0777); // Create output folder

        // TODO: Memcopy from device to host
        writeDataVTK(outputName, h_phi, h_curvature, h_u, h_v, nx, ny, dx, dy, count++);
    }

    MPI_Bcast(&arraySplittedSize, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    printf("Ahahah the alloc is : %ld\n", arraySplittedSize);

    double *h_curvature_splitted = new double[arraySplittedSize];
    double *h_lengths_splitted = new double[arraySplittedSize];

    // Loop over time
    for (int step = 1; step <= nSteps; step++)
    {
        double max = 0;
        double total_length = 0;

        if (world_rank == 0)
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
        }
        MPI_Scatter(h_curvature, arraySplittedSize, MPI_DOUBLE, h_curvature_splitted, arraySplittedSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Reduce(h_curvature, &max, arraySplittedSize, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(h_lengths, &total_length, arraySplittedSize, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        // MPI_Scatter(h_phi, recvcount, MPI_DOUBLE, h_phi, recvcount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // MPI_Scatter(h_curvature, recvcount, MPI_DOUBLE, h_curvature, recvcount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // MPI_Scatter(h_u, recvcount, MPI_DOUBLE, h_u, recvcount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // MPI_Scatter(h_v, recvcount, MPI_DOUBLE, h_v, recvcount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // MPI_Bcast(&max, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // MPI_Bcast(&total_length, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Write data to output file
        if (world_rank == 0 && step % outputFrequency == 0)
        {
            cout << "Step: " << step << "\n\n";
            writeDataVTK(outputName, h_phi, h_curvature, h_u, h_v, nx, ny, dx, dy, count++);
        }
    }

    if (world_rank == 0)
    {
        // Free memory
        delete[] h_phi, h_curvature, h_u, h_v;

        CHECK_ERROR(cudaFree((void **)d_phi));
        CHECK_ERROR(cudaFree((void **)d_phi_n));
        CHECK_ERROR(cudaFree((void **)d_curvature));
        CHECK_ERROR(cudaFree((void **)d_u));
        CHECK_ERROR(cudaFree((void **)d_v));
    }

    return 0;
}
