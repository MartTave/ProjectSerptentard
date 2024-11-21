// Libraries
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <chrono>

// == User lib ==
#include "diagnostics/diagnostics.h"
#include "initialization/init.h"
#include "solve/solve.h"
#include "write/write.h"

// Namespace
using namespace std;

// Advection Solver
int main(int argc, char *argv[])
{

    auto startInit = chrono::high_resolution_clock::now();

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
    double **phi = new double *[nx];       // LevelSet field
    double **curvature = new double *[nx]; // Curvature field
    double **u = new double *[nx];         // Velocity field in x-direction
    double **v = new double *[nx];         // Velocity field in y-direction
    for (int i = 0; i < nx; ++i)
    {
        phi[i] = new double[ny];
        curvature[i] = new double[ny];
        u[i] = new double[ny];
        v[i] = new double[ny];
    }

    Initialization(phi, curvature, u, v, nx, ny, dx, dy); // Initialize the distance function field
    computeBoundaries(phi, nx, ny);                       // Extrapolate phi on the boundaries

    // == Output ==
    stringstream ss;
    ss << scale;
    string scaleStr = ss.str();

    string outputName = "output/levelSet_scale" + scaleStr + "_";
    int count = 0; // Number of VTK file already written

    // == First output ==
    // Write data in VTK format
    mkdir("output", 0777); // Create output folder
    writeDataVTK(outputName, phi, curvature, u, v, nx, ny, dx, dy, count++);

    auto endInit = chrono::high_resolution_clock::now();

    // Loop over time
    for (int step = 1; step <= nSteps; step++)
    {

        time += dt; // Simulation time increases

        // Solve the advection equation
        solveAdvectionEquationExplicit(phi, u, v, nx, ny, dx, dy, dt);

        // Diagnostics: interface perimeter
        computeInterfaceLength(phi, nx, ny, dx, dy);

        // Diagnostics: interface curvature
        computeInterfaceCurvature(phi, curvature, nx, ny, dx, dy);

        // Write data to output file
        if (step % outputFrequency == 0)
        {
            cout << "\nStarting iteration step " << step << "/" << nSteps << "\tTime " << time << "s\n";
            writeDataVTK(outputName, phi, curvature, u, v, nx, ny, dx, dy, count++);
        }
    }

    auto endLoop = chrono::high_resolution_clock::now();

    // Deallocate memory
    for (int i = 0; i < nx; ++i)
    {
        delete[] phi[i];
    }
    delete[] phi;

    auto endDeallocate = chrono::high_resolution_clock::now();

    int initDuration = chrono::duration_cast<chrono::milliseconds>(endInit - startInit).count();
    int loopDuration = chrono::duration_cast<chrono::milliseconds>(endLoop - endInit).count();
    int deallocateDuration = chrono::duration_cast<chrono::milliseconds>(endDeallocate - endLoop).count();
    int totalDuration = chrono::duration_cast<chrono::milliseconds>(endDeallocate - startInit).count();

    cout << "Sooooo, actually :\n";
    cout << "Initialization took " << initDuration << "ms\n";
    cout << "Loop took " << loopDuration << "ms\n";
    cout << "Deallocate took " << deallocateDuration << "ms\n";
    cout << "Total took " << totalDuration << "ms\n";
    cout << "For scale " << scale << "\n";

    return 0;
}
