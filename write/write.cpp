#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <mpi.h>
#include "write.h"

#include <bits/stdc++.h>

using namespace std;

// Write data to VTK file
void writeDataVTK(const string filename, double *phi, double *curvature, double *u, double *v, const int nx, const int ny, const double dx, const double dy, const int step)
{

    // Create the filename
    string filename_all = "0000000" + to_string(step);
    reverse(filename_all.begin(), filename_all.end());
    filename_all.resize(7);
    reverse(filename_all.begin(), filename_all.end());
    filename_all = filename + filename_all + ".vtk";

    // Inform user the output filename
    cout << "Writing data into " << filename_all << "\n";

    // Setting open file using output file streaming
    ofstream myfile;
    myfile.open(filename_all);

    // Write header of vtk file
    myfile << "# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET RECTILINEAR_GRID\n";

    // Write domain dimensions (must be 3D)
    myfile << "DIMENSIONS " << nx << " " << ny << " 1\n";
    myfile << "X_COORDINATES " << nx << " float\n";
    for (int i = 0; i < nx; i++)
    {
        myfile << i * dx << "\n";
    }

    myfile << "Y_COORDINATES " << ny << " float\n";
    for (int j = 0; j < ny; j++)
    {
        myfile << j * dy << "\n";
    }

    myfile << "Z_COORDINATES 1 float\n";
    myfile << "0\n";

    // Write number of cells
    myfile << "POINT_DATA " << nx * ny << "\n";

    // Write the phi values (loop over ny then nx)
    myfile << "SCALARS phi float 1\n";
    myfile << "LOOKUP_TABLE default\n";

    for (int j = 0; j < ny; j++)
    {
        for (int i = 0; i < nx; i++)
        {
            myfile << phi[i * ny + j] << "\n";
        }
    }

    // Write the x velocity values (loop over ny then nx)
    myfile << "\nSCALARS u float 1\n";
    myfile << "LOOKUP_TABLE default\n";

    for (int j = 0; j < ny; j++)
    {
        for (int i = 0; i < nx; i++)
        {
            myfile << u[i * ny + j] << "\n";
        }
    }

    // Write the y velocity values (loop over ny then nx)
    myfile << "\nSCALARS v float 1\n";
    myfile << "LOOKUP_TABLE default\n";

    for (int j = 0; j < ny; j++)
    {
        for (int i = 0; i < nx; i++)
        {
            myfile << v[i * ny + j] << "\n";
        }
    }

    // Write the curvature values (loop over ny then nx)
    myfile << "\nSCALARS curvature float 1\n";
    myfile << "LOOKUP_TABLE default\n";

    for (int j = 0; j < ny; j++)
    {
        for (int i = 0; i < nx; i++)
        {
            myfile << curvature[i * ny + j] << "\n";
        }
    }

    // Close file
    myfile.close();
}

// void writeDataVTKv2(const string filename, double *phi, double *curvature, double *u, double *v, const int nx, const int ny, const double dx, const double dy, const int step, const int world_rank)
// {
//     if (world_rank == 0)
//     {
//         size("asdasdas");
//         sizeof("asdadasd") / sizeof(string);
//         // Create the filename
//         string filename_all = "0000000" + to_string(step);
//         reverse(filename_all.begin(), filename_all.end());
//         filename_all.resize(7);
//         reverse(filename_all.begin(), filename_all.end());
//         filename_all = filename + filename_all + ".vtk";

//         // Inform user the output filename
//         cout << "Writing data into " << filename_all << "\n";

//         // Setting open file using output file streaming
//         ofstream myfile;
//         myfile.open(filename_all);

//         // Write header of vtk file
//         myfile << "# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET RECTILINEAR_GRID\n";

//         // Write domain dimensions (must be 3D)
//         myfile << "DIMENSIONS " << nx << " " << ny << " 1\n";
//         myfile << "X_COORDINATES " << nx << " float\n";
//     }

//     for (int i = 0; i < nx; i++)
//     {
//         myfile << i * dx << "\n";
//     }

//     myfile << "Y_COORDINATES " << ny << " float\n";
//     for (int j = 0; j < ny; j++)
//     {
//         myfile << j * dy << "\n";
//     }

//     myfile << "Z_COORDINATES 1 float\n";
//     myfile << "0\n";

//     // Write number of cells
//     myfile << "POINT_DATA " << nx * ny << "\n";

//     // Write the phi values (loop over ny then nx)
//     myfile << "SCALARS phi float 1\n";
//     myfile << "LOOKUP_TABLE default\n";

//     for (int j = 0; j < ny; j++)
//     {
//         for (int i = 0; i < nx; i++)
//         {
//             myfile << phi[i * ny + j] << "\n";
//         }
//     }

//     // Write the x velocity values (loop over ny then nx)
//     myfile << "\nSCALARS u float 1\n";
//     myfile << "LOOKUP_TABLE default\n";

//     for (int j = 0; j < ny; j++)
//     {
//         for (int i = 0; i < nx; i++)
//         {
//             myfile << u[i * ny + j] << "\n";
//         }
//     }

//     // Write the y velocity values (loop over ny then nx)
//     myfile << "\nSCALARS v float 1\n";
//     myfile << "LOOKUP_TABLE default\n";

//     for (int j = 0; j < ny; j++)
//     {
//         for (int i = 0; i < nx; i++)
//         {
//             myfile << v[i * ny + j] << "\n";
//         }
//     }

//     // Write the curvature values (loop over ny then nx)
//     myfile << "\nSCALARS curvature float 1\n";
//     myfile << "LOOKUP_TABLE default\n";

//     for (int j = 0; j < ny; j++)
//     {
//         for (int i = 0; i < nx; i++)
//         {
//             myfile << curvature[i * ny + j] << "\n";
//         }
//     }

//     // Close file
//     myfile.close();
// }

// void writeDataMPIVTK(const string filename, double *phi, double *curvature, double *u, double *v, const int nx, const int ny, const double dx, const double dy, const int step)
// {

//     MPI_File fh;
//     int rank;
//     int n_pro=72;
//     int elem_pr_proc=sizeof(curvature)/72;

//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     double *part[sizeof(double)*elem_pr_proc];

//     // Create the filename
//     string filename_all = "0000000" + to_string(step);
//     reverse(filename_all.begin(), filename_all.end());
//     filename_all.resize(7);
//     reverse(filename_all.begin(), filename_all.end());
//     filename_all = filename + filename_all + ".vtk";
//     if (rank == 0)
//         // Inform user the output filename
//         cout << "Writing data into " << filename_all << "\n";

//     // Setting open file using output file streaming
//     //ofstream myfile;
//     //myfile.open(filename_all);
//     char filenameCH[sizeof(filename_all) + 1];
//     strcpy(filenameCH, filename_all.c_str());
//     MPI_File_open(MPI_COMM_WORLD, filenameCH,
//         MPI_MODE_CREATE|MPI_MODE_WRONLY,
//         MPI_INFO_NULL, &fh);

//     if (rank == 0)
//     {
//         char txt[]="# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET RECTILINEAR_GRID\n";
//         MPI_File_write_all(fh, txt, sizeof(txt), MPI_CHAR, MPI_STATUS_IGNORE);
//         string text="DIMENSIONS " + to_string(nx) + " " + to_string(ny) + " 1\n";
//         char txt[sizeof(text)+1];
//         strcpy(txt, text.c_str());
//         MPI_File_write_all(fh, txt, sizeof(txt), MPI_CHAR, MPI_STATUS_IGNORE);
//         string text="X_COORDINATES " + to_string(nx) + " float\n";
//         char txt[sizeof(text)+1];
//         strcpy(txt, text.c_str());
//         MPI_File_write_all(fh, txt, sizeof(txt), MPI_CHAR, MPI_STATUS_IGNORE);

//         string out;
//         for (int i = 0; i < nx; i++)
//         {
//             out += to_string(i * dx) + "\n";
//         }
//         char arr[sizeof(out) + 1];

//         // copying the contents of the string to
//         // char array
//         strcpy(arr, out.c_str());
//         MPI_File_write_all(fh, arr, sizeof(arr), MPI_CHAR, MPI_STATUS_IGNORE);

//         string text="Y_COORDINATES " + to_string(nx) + " float\n";
//         char txt[sizeof(text)+1];
//         strcpy(txt, text.c_str());
//         MPI_File_write_all(fh, txt, sizeof(txt), MPI_CHAR, MPI_STATUS_IGNORE);
//         out="";
//         for (int j = 0; j < ny; j++)
//         {
//             out += to_string(j * dy) + "\n";
//         }
//         char arr[sizeof(out) + 1];

//         // copying the contents of the string to
//         // char array
//         strcpy(arr, out.c_str());
//         MPI_File_write_all(fh, arr, sizeof(arr), MPI_CHAR, MPI_STATUS_IGNORE);

//         string text = "Z_COORDINATES 1 float\n0\nPOINT_DATA " + to_string(nx * ny) + "\nSCALARS phi float 1\nLOOKUP_TABLE default\n";
//         char txt[sizeof(text)+1];
//         strcpy(txt, text.c_str());
//         MPI_File_write_all(fh, txt, sizeof(txt), MPI_CHAR, MPI_STATUS_IGNORE);
//     }
//     MPI_Barrier(MPI_COMM_WORLD);

//     if(rank==0){
//         MPI_Scatter(curvature, elem_pr_proc, MPI_DOUBLE, part,
//             elem_pr_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);

//     }
//     MPI_Barrier(MPI_COMM_WORLD);
//     MPI_Gather(&part, 1, MPI_DOUBLE, part, 1, MPI_DOUBLE, 0,
//            MPI_COMM_WORLD);

//     MPI_File_write_all(fh, part, sizeof(part), MPI_DOUBLE, MPI_STATUS_IGNORE);
//     MPI_Barrier(MPI_COMM_WORLD);
//     if(rank==0){
//         MPI_Scatter(phi, elem_pr_proc, MPI_DOUBLE, part,
//             elem_pr_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);

//     }
//     MPI_Barrier(MPI_COMM_WORLD);
//     MPI_Gather(&part, 1, MPI_DOUBLE, part, 1, MPI_DOUBLE, 0,
//            MPI_COMM_WORLD);

//     MPI_File_write_all(fh, part, sizeof(part), MPI_DOUBLE, MPI_STATUS_IGNORE);
//     MPI_Barrier(MPI_COMM_WORLD);

//     if(rank==0){
//         char txt[]="\nSCALARS u float 1\nLOOKUP_TABLE default\n";
//         MPI_File_write_all(fh, txt,sizeof(txt), MPI_CHAR, MPI_STATUS_IGNORE);
//     }

//     if(rank==0){
//         MPI_Scatter(u, elem_pr_proc, MPI_DOUBLE, part,
//             elem_pr_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);

//     }
//     MPI_Barrier(MPI_COMM_WORLD);
//     MPI_Gather(&part, 1, MPI_DOUBLE, part, 1, MPI_DOUBLE, 0,
//            MPI_COMM_WORLD);

//     MPI_File_write_all(fh, part, sizeof(part), MPI_DOUBLE, MPI_STATUS_IGNORE);
//     MPI_Barrier(MPI_COMM_WORLD);

//     if(rank==0){
//         char txt[]="\nSCALARS v float 1\nLOOKUP_TABLE default\n";
//         MPI_File_write_all(fh, txt,sizeof(txt), MPI_CHAR, MPI_STATUS_IGNORE);
//     }

//     if(rank==0){
//         MPI_Scatter(v, elem_pr_proc, MPI_DOUBLE, part,
//             elem_pr_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);

//     }
//     MPI_Barrier(MPI_COMM_WORLD);
//     MPI_Gather(&part, 1, MPI_DOUBLE, part, 1, MPI_DOUBLE, 0,
//            MPI_COMM_WORLD);

//     MPI_File_write_all(fh, part, sizeof(part), MPI_DOUBLE, MPI_STATUS_IGNORE);
//     MPI_Barrier(MPI_COMM_WORLD);

//     if(rank==0){
//         char txt[]="\nSCALARS curvature float 1\nLOOKUP_TABLE default\n";
//         MPI_File_write_all(fh, txt,sizeof(txt), MPI_CHAR, MPI_STATUS_IGNORE);
//     }

//     if(rank==0){
//         MPI_Scatter(curvature, elem_pr_proc, MPI_DOUBLE, part,
//             elem_pr_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);

//     }
//     MPI_Barrier(MPI_COMM_WORLD);
//     MPI_Gather(&part, 1, MPI_DOUBLE, part, 1, MPI_DOUBLE, 0,
//            MPI_COMM_WORLD);

//     MPI_File_write_all(fh, part, sizeof(part), MPI_DOUBLE, MPI_STATUS_IGNORE);
//     MPI_Barrier(MPI_COMM_WORLD);

//     // Close file
//     //myfile.close();
//     MPI_File_close(&fh);
// }
