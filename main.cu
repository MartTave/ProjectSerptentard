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

    // == MPI Initialization ==
    MPI_Status status;
    int world_size, world_rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    auto initStart = high_resolution_clock::now();

    // Variables declaration
    int scale = 10;
    if (argc > 1)
    {
        scale = stoi(argv[1]);
    }
    int nx = 100 * scale; // Number of cells in each direction
    int ny = 100 * scale; // Number of cells in each direction

    int count = 0; // Number of VTK file already written

    // == Output ==
    string outputName = "output/levelSet_scale" + to_string(scale) + "_";

    // == Host Variables ==
    long arrayLength = nx * ny;
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

    // == Spatial ==
    double Lx = 1.0;           // Square domain [m]
    double Ly = 1.0;           // Square domain [m]
    double dx = Lx / (nx - 1); // Spatial step [m]
    double dy = Ly / (ny - 1); // Spatial step [m]

    // == Temporal ==
    double tFinal = 4.0;           // Final time [s]
    double dt = 0.005 / scale;     // Temporal step [s]
    int nSteps = int(tFinal / dt); // Number of steps to perform
    double time = 0.0;             // Actual Simulation time [s]

    // == Numerical ==
    int outputFrequency = nSteps / 40;

    int windowSize = 25;
    int gridWidth = (nx + windowSize - 1) / windowSize;
    gridHeight = (ny + windowSize - 1) / windowSize;
    dim3 dimGrid = dim3(gridWidth, gridHeight);
    dim3 dimBlock = dim3(windowSize, windowSize);

    if (world_rank == 0)
    {
        // The rest to allocate to the first cores
        // If rest == 4 -> the first 4 cores will have one element more in their array
        int rest = arrayLength % world_size;
        // This is the base number of elements that each core will have
        int nbrOfElements = arrayLength / world_size;
        for (int i = 0; i < world_size; i++)
        {
            if (i < rest)
            {
                // Here we are at the first cores
                // They need 1 element more in they array to make the array split work
                arrStart[i] = i * (nbrOfElements + 1);
                arrEnd[i] = (i + 1) * (nbrOfElements + 1);
                splittedLengthes[i] = (nbrOfElements + 1);
            }
            else
            {
                // Here we are at the last cores
                // They can take the right number of element
                arrStart[i] = rest * (nbrOfElements + 1) + (i - rest) * nbrOfElements;
                arrEnd[i] = rest * (nbrOfElements + 1) + (i - rest + 1) * nbrOfElements;
                splittedLengthes[i] = nbrOfElements;
            }
            // This is not really necessary, but as we will not have that many cores, it will not be a problem
            splittedSizes[i] = splittedLengthes[i] * sizeof(double);
        }
    }

    // Casting sizes calculation to everyone
    MPI_Bcast(arrStart, world_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(arrEnd, world_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(splittedLengthes, world_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(splittedSizes, world_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Host copy of CUDA Device arrays
    double *h_phi_splitted = new double[splittedLengthes[world_rank]];
    double *h_curvature_splitted = new double[splittedLengthes[world_rank]];
    double *h_lengths_splitted = new double[splittedLengthes[world_rank]];
    double *h_u_splitted = new double[splittedLengthes[world_rank]];
    double *h_v_splitted = new double[splittedLengthes[world_rank]];

    // CUDA Device pointer (arrrays)
    double *d_phi;
    double *d_phi_n;
    double *d_curvature;
    double *d_lengths;
    double *d_u;
    double *d_v;

    if (world_rank == 0)
    {
        // Allocating everything on the CUDA device and creating the result folder
        mkdir("output", cudaMalloc 0777); // Create output folder
        CHECK_ERROR(((void **)&d_phi, size));
        CHECK_ERROR(cudaMalloc((void **)&d_lengths, size));
        CHECK_ERROR(cudaMalloc((void **)&d_phi_n, size));
        CHECK_ERROR(cudaMalloc((void **)&d_curvature, size));
        CHECK_ERROR(cudaMalloc((void **)&d_u, size));
        CHECK_ERROR(cudaMalloc((void **)&d_v, size));

        // Launching the initialization kernel
        InitializationKernel<<<dimGrid, dimBlock>>>(d_phi, d_curvature, d_u, d_v, nx, ny, dx, dy);
        // Waiting for the kernel to finish
        cudaDeviceSynchronize();
        // Launching the boundaires kernels
        computeBoundariesLines<<<1, nx>>>(d_phi, nx, ny);
        computeBoundariesColumns<<<1, ny>>>(d_phi, nx, ny);
        // Waiting for the kernel to finish
        cudaDeviceSynchronize();
    }

    if (world_rank == 0)
    {
        // Getting the results from the device
        CHECK_ERROR(cudaMemcpy(h_phi, d_phi, size, cudaMemcpyDeviceToHost));
        CHECK_ERROR(cudaMemcpy(h_curvature, d_curvature, size, cudaMemcpyDeviceToHost));
        CHECK_ERROR(cudaMemcpy(h_u, d_u, size, cudaMemcpyDeviceToHost));
        CHECK_ERROR(cudaMemcpy(h_v, d_v, size, cudaMemcpyDeviceToHost));
        for (int i = 1; i < world_size; i++)
        {
            // Core 0 will send parts of the arrays to each core
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
        // Every other core will receive the parts of the arrays from the core 0
        MPI_Recv(h_phi_splitted, splittedLengthes[world_rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(h_curvature_splitted, splittedLengthes[world_rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(h_u_splitted, splittedLengthes[world_rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(h_v_splitted, splittedLengthes[world_rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    }

    // Then  every core can transform their array in string to write it in the VTK file
    string toWriteU = getString(h_u_splitted, splittedLengthes[world_rank], world_rank);
    string toWriteV = getString(h_v_splitted, splittedLengthes[world_rank], world_rank);
    string toWritePhi = getString(h_phi_splitted, splittedLengthes[world_rank], world_rank);
    string toWriteCurvature = getString(h_curvature_splitted, splittedLengthes[world_rank], world_rank);
    // Launching write funtion with each part of the data to write
    writeDataVTK(outputName, toWritePhi, toWriteCurvature, toWriteU, toWriteV, nx, ny, dx, dy, count++, world_rank, world_size);

    auto initEnd = high_resolution_clock::now();
    // Loop over time
    for (int step = 1; step <= nSteps; step++)
    {
        // Resetting variables
        double max = 0;
        double total_length = 0;

        if (world_rank == 0)
        {

            time += dt; // Simulation time increases

            // Solve the advection equation
            copyPhi<<<dimGrid, dimBlock>>>(d_phi, d_phi_n, nx, ny);
            solveAdvectionEquationExplicit<<<dimGrid, dimBlock>>>(d_phi, d_phi_n, d_u, d_v, nx, ny, dx, dy, dt);

            // Waiting for kernels to finish
            cudaDeviceSynchronize();

            // Computing boundaries
            computeBoundariesLines<<<1, nx>>>(d_phi, nx, ny);
            computeBoundariesColumns<<<1, ny>>>(d_phi, nx, ny);

            // Waiting
            cudaDeviceSynchronize();

            // Diagnostics: interface perimeter
            computeInterfaceLengthKernel<<<dimGrid, dimBlock>>>(d_phi, d_lengths, nx, ny, dx, dy);

            // Diagnostics: interface curvature
            computeInterfaceCurvatureKernel<<<dimGrid, dimBlock>>>(d_phi, d_curvature, nx, ny, dx, dy);

            cudaDeviceSynchronize();
            // Copying results to CPU memory
            CHECK_ERROR(cudaMemcpy(h_phi, d_phi, size, cudaMemcpyDeviceToHost));
            CHECK_ERROR(cudaMemcpy(h_curvature, d_curvature, size, cudaMemcpyDeviceToHost));
            CHECK_ERROR(cudaMemcpy(h_lengths, d_lengths, size, cudaMemcpyDeviceToHost));

            for (int i = 1; i < world_size; i++)
            {
                // Sending results part to each core
                MPI_Send(h_phi + arrStart[i], splittedLengthes[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                MPI_Send(h_curvature + arrStart[i], splittedLengthes[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                MPI_Send(h_lengths + arrStart[i], splittedLengthes[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            }
            // TODO: Maybe we don't need to do that, as the pointer should stay at the right place
            h_phi_splitted = h_phi;
            h_curvature_splitted = h_curvature;
            h_lengths_splitted = h_lengths;
        }
        else
        {
            // Receiving the results from the core 0
            MPI_Recv(h_phi_splitted, splittedLengthes[world_rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(h_curvature_splitted, splittedLengthes[world_rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(h_lengths_splitted, splittedLengthes[world_rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        }

        // Every core computes their local sum and max
        double localSum = 0;
        double localMax = 0;
        for (int i = 0; i < splittedLengthes[world_rank]; i++)
        {
            localSum += h_lengths_splitted[i];
            if (abs(h_curvature_splitted[i]) > localMax)
            {
                localMax = abs(h_curvature_splitted[i]);
            }
        }
        // Reducing the results to core 0
        MPI_Reduce(&localMax, &max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localSum, &total_length, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // Write data to output file i needed
        if (step % outputFrequency == 0)
        {
            // Every core transforms their array in string to write it in the VTK file
            string toWritePhi = getString(h_phi_splitted, splittedLengthes[world_rank], world_rank);
            string toWriteCurvature = getString(h_curvature_splitted, splittedLengthes[world_rank], world_rank);
            writeDataVTK(outputName, toWritePhi, toWriteCurvature, toWriteU, toWriteU, nx, ny, dx, dy, count++, world_rank, world_size);
            if (world_rank == 0)
            {
                cout << "Step: " << step << "\n\n";
            }
        }
    }
    auto loopEnd = high_resolution_clock::now();
    delete[] h_phi, h_curvature, h_u, h_v;

    if (world_rank == 0)
    {
        // Free memory

        CHECK_ERROR(cudaFree((void **)d_phi));
        CHECK_ERROR(cudaFree((void **)d_phi_n));
        CHECK_ERROR(cudaFree((void **)d_curvature));
        CHECK_ERROR(cudaFree((void **)d_u));
        CHECK_ERROR(cudaFree((void **)d_v));
    }

    if (world_rank == 0)
    {
        auto end = high_resolution_clock::now();
        int initDuration = chrono::duration_cast<chrono::milliseconds>(initEnd - initStart).count();
        int loopDuration = chrono::duration_cast<chrono::milliseconds>(loopEnd - initEnd).count();
        int totalDuration = chrono::duration_cast<chrono::milliseconds>(end - initStart).count();
        int deallocateDuration = chrono::duration_cast<chrono::milliseconds>(end - loopEnd).count();

        cout << "Sooooo, actually :\n";
        cout << "Initialization took " << initDuration << "ms\n";
        cout << "Loop took " << loopDuration << "ms\n";
        cout << "Deallocate took " << deallocateDuration << "ms\n";
        cout << "Total took " << totalDuration << "ms\n";
        cout << "For scale " << scale << "\n";
    }
    MPI_Finalize();
    return 0;
}
