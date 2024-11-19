// Libraries
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <cuda.h>
#include <mpi.h>
#include <chrono>

// == User lib ==
#include "diagnostics/diagnostics.cuh"
#include "initialization/init.cuh"
#include "solve/solve.cuh"
#include "write/write.h"

#include "common_includes.cu"

// Namespace
using namespace std;
using namespace std::chrono;

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

    long sum = 0;

    double Lx, Ly, dx, dy, tFinal, dt, time;

    long arrayLength, arraySplittedSize;
    arrayLength = nx * ny;
    stringstream ss;

    int count = 0; // Number of VTK file already written
    string scaleStr = ss.str();
    // == Output ==
    ss << scale;
    string outputName = "output/levelSet_scale" + scaleStr + "_";

    dim3 dimGrid, dimBlock;

    double *h_phi = new double[arrayLength];
    double *h_curvature = new double[arrayLength + (arrayLength % world_size)];
    double *h_u = new double[arrayLength + (arrayLength % world_size)];
    double *h_v = new double[arrayLength + (arrayLength % world_size)];
    double *h_lengths = new double[arrayLength + (arrayLength % world_size)];
    long size = arrayLength * sizeof(double);
    int *arrStart = new int[world_size];
    int *arrEnd = new int[world_size];
    int *splittedLengthes = new int[world_size];
    int *splittedSizes = new int[world_size];

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

    windowSize = 25;
    gridWidth = (nx + windowSize - 1) / windowSize;
    gridHeight = (ny + windowSize - 1) / windowSize;
    dimGrid = dim3(gridWidth, gridHeight);
    dimBlock = dim3(windowSize, windowSize);

    if (world_rank == 0)
    {
        int rest = arrayLength % world_size;
        int nbrOfElements = arrayLength / world_size;
        for (int i = 0; i < world_size; i++)
        {
            if (i < rest)
            {
                arrStart[i] = i * (nbrOfElements + 1);
                arrEnd[i] = (i + 1) * (nbrOfElements + 1);
                splittedLengthes[i] = (nbrOfElements + 1);
            }
            else
            {
                arrStart[i] = rest * (nbrOfElements + 1) + (i - rest) * nbrOfElements;
                arrEnd[i] = rest * (nbrOfElements + 1) + (i - rest + 1) * nbrOfElements;
                splittedLengthes[i] = nbrOfElements;
            }
            splittedSizes[i] = splittedLengthes[i] * sizeof(double);
            printf("Array start and end is : %d %d\n", arrStart[i], arrEnd[i]);
        }
    }

    MPI_Bcast(arrStart, world_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(arrEnd, world_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(splittedLengthes, world_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(splittedSizes, world_size, MPI_INT, 0, MPI_COMM_WORLD);

    double *h_phi_splitted = new double[splittedLengthes[world_rank]];
    double *h_curvature_splitted = new double[splittedLengthes[world_rank]];
    double *h_lengths_splitted = new double[splittedLengthes[world_rank]];
    double *h_u_splitted = new double[splittedLengthes[world_rank]];
    double *h_v_splitted = new double[splittedLengthes[world_rank]];

    double *d_phi;
    double *d_phi_n;
    double *d_curvature;
    double *d_lengths;
    double *d_u;
    double *d_v;

    if (world_rank == 0)
    {
        mkdir("output", 0777); // Create output folder
        CHECK_ERROR(cudaMalloc((void **)&d_phi, size));
        CHECK_ERROR(cudaMalloc((void **)&d_lengths, size));
        CHECK_ERROR(cudaMalloc((void **)&d_phi_n, size));
        CHECK_ERROR(cudaMalloc((void **)&d_curvature, size));
        CHECK_ERROR(cudaMalloc((void **)&d_u, size));
        CHECK_ERROR(cudaMalloc((void **)&d_v, size));

        InitializationKernel<<<dimGrid, dimBlock>>>(d_phi, d_curvature, d_u, d_v, nx, ny, dx, dy);
        cudaDeviceSynchronize();
        computeBoundariesLines<<<1, nx>>>(d_phi, nx, ny);
        computeBoundariesColumns<<<1, ny>>>(d_phi, nx, ny);
        cudaDeviceSynchronize();
    }

    size_t pointerSize = sizeof(void *);

    if (world_rank == 0)
    {
        CHECK_ERROR(cudaMemcpy(h_phi, d_phi, size, cudaMemcpyDeviceToHost));
        CHECK_ERROR(cudaMemcpy(h_curvature, d_curvature, size, cudaMemcpyDeviceToHost));
        CHECK_ERROR(cudaMemcpy(h_u, d_u, size, cudaMemcpyDeviceToHost));
        CHECK_ERROR(cudaMemcpy(h_v, d_v, size, cudaMemcpyDeviceToHost));
        for (int i = 1; i < world_size; i++)
        {
            MPI_Send(h_phi + arrStart[i], splittedLengthes[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            MPI_Send(h_curvature + arrStart[i], splittedLengthes[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            MPI_Send(h_u + arrStart[i], splittedLengthes[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            MPI_Send(h_v + arrStart[i], splittedLengthes[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }

        // We can set the starting pointer to the first element of the array, as the size will delimit what will be sent to the next function
        h_phi_splitted = h_phi;
        h_curvature_splitted = h_curvature;
        h_u_splitted = h_u;
        h_v_splitted = h_v;
    }
    else
    {
        MPI_Recv(h_phi_splitted, splittedLengthes[world_rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(h_curvature_splitted, splittedLengthes[world_rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(h_u_splitted, splittedLengthes[world_rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(h_v_splitted, splittedLengthes[world_rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    }

    string toWriteU = getString(h_u_splitted, splittedLengthes[world_rank]);
    string toWriteV = getString(h_v_splitted, splittedLengthes[world_rank]);
    string toWritePhi = getString(h_phi_splitted, splittedLengthes[world_rank]);
    string toWriteCurvature = getString(h_curvature_splitted, splittedLengthes[world_rank]);
    if (world_rank == 0) {
	cout << "Writing initial data\n";
    }
    writeDataVTK(outputName, toWritePhi, toWriteCurvature, toWriteU, toWriteV, nx, ny, dx, dy, count++, world_rank);
    if (world_rank == 0) {
	cout << "Done - Written : " << outputName << "\n";
    }
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

            cudaDeviceSynchronize();
            CHECK_ERROR(cudaMemcpy(h_phi, d_phi, size, cudaMemcpyDeviceToHost));
            CHECK_ERROR(cudaMemcpy(h_curvature, d_curvature, size, cudaMemcpyDeviceToHost));
            CHECK_ERROR(cudaMemcpy(h_lengths, d_lengths, size, cudaMemcpyDeviceToHost));

            for (int i = 1; i < world_size; i++)
            {
                MPI_Send(h_phi + arrStart[i], splittedLengthes[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                MPI_Send(h_curvature + arrStart[i], splittedLengthes[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                MPI_Send(h_lengths + arrStart[i], splittedLengthes[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            }
            // TODO:Maybe no need to do this, as the pointer still point to the same place
            h_phi_splitted = h_phi;
            h_curvature_splitted = h_curvature;
            h_lengths_splitted = h_lengths;
            cout << "Done loop\n";
        }
        else
        {
            MPI_Recv(h_phi_splitted, splittedLengthes[world_rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(h_curvature_splitted, splittedLengthes[world_rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(h_lengths_splitted, splittedLengthes[world_rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        }

        double localSum = 0;
        double localMax = 0;
        for (int i = 0; i < arraySplittedSize; i++)
        {
            localSum += h_lengths_splitted[i];
            if (abs(h_curvature_splitted[i]) > localMax)
            {
                localMax = abs(h_curvature_splitted[i]);
            }
        }
        MPI_Reduce(&localMax, &max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localSum, &total_length, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (world_rank == 0)
        {
            cout << "Done reduce\n";
        }

        string toWritePhi = getString(h_phi_splitted, splittedLengthes[world_rank]);
        string toWriteCurvature = getString(h_curvature_splitted, splittedLengthes[world_rank]);

        writeDataVTK(outputName, toWritePhi, toWriteCurvature, toWriteU, toWriteU, nx, ny, dx, dy, count++, world_rank);

        // Write data to output file
        if (world_rank == 0 && step % outputFrequency == 0)
        {
            cout << "Step: " << step << "\n\n";
        }
    }

    delete[] h_phi, h_curvature, h_u, h_v;

    if (world_rank == 0)
    {
        // Free memory

        CHECK_ERROR(cudaFree((void **)d_phi));
        CHECK_ERROR(cudaFree((void **)d_phi_n));
        CHECK_ERROR(cudaFree((void **)d_curvature));
        CHECK_ERROR(cudaFree((void **)d_u));
        CHECK_ERROR(cudaFree((void **)d_v));
        printf("File writing took : %ld ns\n", sum);
    }
    MPI_Finalize();
    return 0;
}
