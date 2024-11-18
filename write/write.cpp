#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <mpi.h>
#include "write.h"

#include <bits/stdc++.h>

#include "../common_includes.cu"

using namespace std;

string getString(double *data, long size)
{
    string toWrite;
    for (int i = 0; i < size; i++)
    {
        toWrite += to_string(data[i]) + "\n";
    }
    return toWrite;
}

// Write data to VTK file
void writeDataVTK(const string filename, string phi_part, string curvature_part, string u_part, string v_part, const int nx, const int ny, const double dx, const double dy, const int step, const int world_rank)
{

    MPI_File fh;
    string filename_all = "0000000" + to_string(step);
    reverse(filename_all.begin(), filename_all.end());
    filename_all.resize(7);
    reverse(filename_all.begin(), filename_all.end());
    filename_all = filename + filename_all + ".vtk";
    MPI_File_open(MPI_COMM_WORLD, filename_all.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    MPI_Offset header_offset;
    if (world_rank == 0)
    {
        string header = "# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET RECTILINEAR_GRID\n";
        header += "DIMENSIONS " + to_string(nx) + " " + to_string(ny) + " 1\n";
        header += "X_COORDINATES " + to_string(nx) + " float\n";
        for (int i = 0; i < nx; i++)
        {
            header += to_string(i * dx) + "\n";
        }
        header += "Y_COORDINATES " + to_string(ny) + " float\n";
        for (int j = 0; j < ny; j++)
        {
            header += to_string(j * dy) + "\n";
        }
        header += "Z_COORDINATES 1 float\n0\nPOINT_DATA " + to_string(nx * ny) + "\n";
        header += "SCALARS phi float 1\nLOOKUP_TABLE default\n";
        MPI_File_write_all(fh, header.c_str(), header.size(), MPI_CHAR, MPI_STATUS_IGNORE);
    }

    // This will sync all cores too !
    MPI_Bcast(&header_offset, 1, MPI_OFFSET, 0, MPI_COMM_WORLD);

    MPI_Offset phi_offset;
    MPI_Offset curvature_offset;
    MPI_Offset u_offset;
    MPI_Offset v_offset;

    long phi_size = phi_part.size() * sizeof(char);
    long curvature_size = curvature_part.size() * sizeof(char);
    long u_size = u_part.size() * sizeof(char);
    long v_size = v_part.size() * sizeof(char);

    MPI_Scan(&phi_size, &phi_offset, 1, MPI_OFFSET, MPI_SUM, MPI_COMM_WORLD);
    MPI_Scan(&curvature_size, &curvature_offset, 1, MPI_OFFSET, MPI_SUM, MPI_COMM_WORLD);
    MPI_Scan(&u_size, &u_offset, 1, MPI_OFFSET, MPI_SUM, MPI_COMM_WORLD);
    MPI_Scan(&v_size, &v_offset, 1, MPI_OFFSET, MPI_SUM, MPI_COMM_WORLD);

    phi_offset = phi_offset - phi_size + header_offset;
    u_offset = u_offset - u_size + phi_offset;
    v_offset = v_offset - v_size + u_offset;
    curvature_offset = curvature_offset - curvature_size + v_offset;

    MPI_File_write_at(fh, phi_offset, phi_part.c_str(), phi_size, MPI_CHAR, MPI_STATUS_IGNORE);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Offset uHeaderSize;
    if (world_rank == 0)
    {
        string uHeader = "\nSCALARS u float 1\nLOOKUP_TABLE default\n";
        uHeaderSize = uHeader.size() * sizeof(char);
        MPI_File_write_at(fh, u_offset, uHeader.c_str(), uHeaderSize, MPI_CHAR, MPI_STATUS_IGNORE);
    }
    MPI_Bcast(&uHeaderSize, 1, MPI_OFFSET, 0, MPI_COMM_WORLD);
    // Offsetting all offsets because we just wrote to file
    u_offset = u_offset + uHeaderSize;
    v_offset += uHeaderSize;
    curvature_offset += uHeaderSize;
    MPI_File_write_at(fh, u_offset, u_part.c_str(), u_size, MPI_CHAR, MPI_STATUS_IGNORE);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Offset vHeaderSize;
    if (world_rank == 0)
    {
        string vHeader = "\nSCALARS v float 1\nLOOKUP_TABLE default\n";
        vHeaderSize = vHeader.size() * sizeof(char);
        MPI_File_write_at(fh, v_offset, vHeader.c_str(), vHeaderSize, MPI_CHAR, MPI_STATUS_IGNORE);
    }
    MPI_Bcast(&vHeaderSize, 1, MPI_OFFSET, 0, MPI_COMM_WORLD);

    v_offset = v_offset + vHeaderSize;
    curvature_offset += vHeaderSize;

    MPI_File_write_at(fh, v_offset, v_part.c_str(), v_size, MPI_CHAR, MPI_STATUS_IGNORE);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Offset curvatureHeaderSize;

    if (world_rank == 0)
    {
        string curvatureHeader = "\nSCALARS curvature float 1\nLOOKUP_TABLE default\n";
        curvatureHeaderSize = curvatureHeader.size() * sizeof(char);
        MPI_File_write_at(fh, curvature_offset, curvatureHeader.c_str(), curvatureHeaderSize, MPI_CHAR, MPI_STATUS_IGNORE);
    }
    MPI_Bcast(&curvatureHeaderSize, 1, MPI_OFFSET, 0, MPI_COMM_WORLD);

    curvature_offset = curvature_offset + curvatureHeaderSize;

    MPI_File_write_at(fh, curvature_offset, curvature_part.c_str(), curvature_size, MPI_CHAR, MPI_STATUS_IGNORE);

    MPI_File_close(&fh);
}

void writeDataVTKv2(const string filename, double *phi, double *curvature, double *u, double *v, const int nx, const int ny, const double dx, const double dy, const int step, const int world_rank)
{
    MPI_File fh;
    string filename_all = "0000000" + to_string(step);
    reverse(filename_all.begin(), filename_all.end());
    filename_all.resize(7);
    reverse(filename_all.begin(), filename_all.end());
    filename_all = filename + filename_all + ".vtk";
    MPI_File_open(MPI_COMM_WORLD, filename_all.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    MPI_Offset base_offset;
    if (world_rank == 0)
    {
        size("asdasdas");
        sizeof("asdadasd") / sizeof(string);
        // Create the filename
        string toWrite = "# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET RECTILINEAR_GRID\n DIMENSIONS " + to_string(nx) + " " + to_string(ny) + " 1\n" + "X_COORDINATES " + to_string(nx) + " float\n";
        base_offset = size(toWrite) * sizeof(char);
        toWrite += "X_COORDINATES " + to_string(ny) + " float\n";
        for (int i = 0; i < nx; i++)
        {
            toWrite += to_string(i * dx) + "\n";
        }
        toWrite += "Y_COORDINATES " + to_string(ny) + " float\n";
        for (int j = 0; j < ny; j++)
        {
            toWrite += to_string(j * dy) + "\n";
        }
        toWrite += "Z_COORDINATES 1 float\n0\nPOINT_DATA " + to_string(nx * ny) + "\n";
        toWrite += "SCALARS phi float 1\nLOOKUP_TABLE default\n";
        base_offset += size(toWrite) * sizeof(char);
        MPI_File_write_at(fh, 0, toWrite.c_str(), base_offset, MPI_CHAR, MPI_STATUS_IGNORE);
    }
    MPI_Bcast(&base_offset, 1, MPI_OFFSET, 0, MPI_COMM_WORLD);

    MPI_Scatter

        //  Each node needs to calculate the size of the data they will write

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
    MPI_File_close(&fh);
}

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
