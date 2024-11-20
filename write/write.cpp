#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <mpi.h>
#include "write.h"

#include <bits/stdc++.h>

using namespace std;

string getString(double *data, long nx, long ny, int world_rank)
{
    string toWrite = "";
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            toWrite += to_string(data[i * ny + j]) + "\n";
        }
    }
    return toWrite;
}

// Write data to VTK file
void writeDataVTK(const string filename, string phi_part, string curvature_part, string u_part, string v_part, const int nx, const int ny, const double dx, const double dy, const int step, const int world_rank, const int world_size)
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
        MPI_File_write(fh, header.c_str(), header.size(), MPI_CHAR, MPI_STATUS_IGNORE);
        header_offset = header.size() * sizeof(char);
    }

    // This will sync all cores too !
    MPI_Bcast(&header_offset, 1, MPI_OFFSET, 0, MPI_COMM_WORLD);
    MPI_Offset phi_offset;
    MPI_Offset curvature_offset;
    MPI_Offset u_offset;
    MPI_Offset v_offset;

    // This is all the sizes for calculated for each core
    long phi_size = phi_part.size() * sizeof(char);
    long curvature_size = curvature_part.size() * sizeof(char);
    long u_size = u_part.size() * sizeof(char);
    long v_size = v_part.size() * sizeof(char);

    // this will sum the offset for each core. So the last core will have all the previous offsets
    MPI_Scan(&phi_size, &phi_offset, 1, MPI_OFFSET, MPI_SUM, MPI_COMM_WORLD);
    MPI_Scan(&curvature_size, &curvature_offset, 1, MPI_OFFSET, MPI_SUM, MPI_COMM_WORLD);
    MPI_Scan(&u_size, &u_offset, 1, MPI_OFFSET, MPI_SUM, MPI_COMM_WORLD);
    MPI_Scan(&v_size, &v_offset, 1, MPI_OFFSET, MPI_SUM, MPI_COMM_WORLD);

    // With those offsets, we now need to add the previous content to those
    phi_offset = phi_offset - phi_size + header_offset;

    MPI_Offset endOfPhi = phi_offset + phi_size;
    MPI_Bcast(&endOfPhi, 1, MPI_OFFSET, world_size - 1, MPI_COMM_WORLD);

    // We then write our first part (for each core)
    MPI_File_write_at(fh, phi_offset, phi_part.c_str(), phi_size, MPI_CHAR, MPI_STATUS_IGNORE); 
    // So we need to sync all cores to be sure they are all done
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Offset uHeaderSize;
    if (world_rank == 0)
    {
        // Then we write the separation (needed by the file format)
        string uHeader = "\nSCALARS u float 1\nLOOKUP_TABLE default\n";
        uHeaderSize = uHeader.size() * sizeof(char);
        MPI_File_write_at(fh, endOfPhi, uHeader.c_str(), uHeaderSize, MPI_CHAR, MPI_STATUS_IGNORE);
    }
    // This will sync all cores
    MPI_Bcast(&uHeaderSize, 1, MPI_OFFSET, 0, MPI_COMM_WORLD);
    // Offsetting all remaining offsets because we just wrote to file
    u_offset = u_offset + uHeaderSize + endOfPhi - u_size;
    MPI_File_write_at(fh, u_offset, u_part.c_str(), u_size, MPI_CHAR, MPI_STATUS_IGNORE);

    MPI_Offset uEnd = u_offset + u_size;

    MPI_Bcast(&uEnd, 1, MPI_OFFSET, world_size - 1, MPI_COMM_WORLD);
 
    MPI_Offset vHeaderSize;
    if (world_rank == 0)
    {
        string vHeader = "\nSCALARS v float 1\nLOOKUP_TABLE default\n";
        vHeaderSize = vHeader.size() * sizeof(char);
        MPI_File_write_at(fh, uEnd, vHeader.c_str(), vHeaderSize, MPI_CHAR, MPI_STATUS_IGNORE);
    }
    MPI_Bcast(&vHeaderSize, 1, MPI_OFFSET, 0, MPI_COMM_WORLD);

    v_offset = v_offset + vHeaderSize + uEnd - v_size;

    MPI_File_write_at(fh, v_offset, v_part.c_str(), v_size, MPI_CHAR, MPI_STATUS_IGNORE);

    MPI_Offset vEnd = v_offset + v_size;

    MPI_Bcast(&vEnd, 1, MPI_OFFSET, world_size - 1, MPI_COMM_WORLD);

    MPI_Offset curvatureHeaderSize;

    if (world_rank == 0)
    {
        string curvatureHeader = "\nSCALARS curvature float 1\nLOOKUP_TABLE default\n";
        curvatureHeaderSize = curvatureHeader.size() * sizeof(char);
        MPI_File_write_at(fh, vEnd, curvatureHeader.c_str(), curvatureHeaderSize, MPI_CHAR, MPI_STATUS_IGNORE);
    }
    MPI_Bcast(&curvatureHeaderSize, 1, MPI_OFFSET, 0, MPI_COMM_WORLD);

    curvature_offset = curvature_offset + curvatureHeaderSize + vEnd - curvature_size;

    MPI_File_write_at(fh, curvature_offset, curvature_part.c_str(), curvature_size, MPI_CHAR, MPI_STATUS_IGNORE);

    MPI_File_close(&fh);
    MPI_Barrier(MPI_COMM_WORLD);
}
