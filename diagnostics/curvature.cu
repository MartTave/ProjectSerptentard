#include <math.h>
#include <iostream>
#include <cuda.h>

void computeInterfaceCurvature(double** phi,double** curvature, const int nx, const int ny, const double dx, const double dy){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    double maxCurvature = 0.0;

    if(i!=0&&j!=0&&i!=nx-1&&j!=ny-1){
        
        if (abs(phi[i+j*nx]) < 3.0 * dx ){ //Compute the curvature only near the interface 

            // first derivative
            double phi_x = (phi[i+1+j*nx]-phi[i-1+j*nx])/2.0/dx;
            double phi_y = (phi[i+(j+1)*nx]-phi[i+(j-1)*nx])/2.0/dy;

            // second derivative
            double phi_xx = (phi[i+1+j*nx]-2.0*phi[i+j*nx]+phi[i-1+j*nx])/dx/dx;
            double phi_yy = (phi[i+(j+1)*nx]-2.0*phi[i+j*nx]+phi[i+(j-1)*nx])/dy/dy;
            double phi_xy = (phi[i+1+(j+1)*nx] - phi[i+1+(j-1)*nx] - phi[i-1+(j+1)*nx] + phi[i-1+(j-1)*nx])/dx/dy/4.0;

            // compute curvature
            curvature[i+j*nx] = (phi_xx*phi_y*phi_y - 2.0* phi_x*phi_y*phi_xy + phi_yy*phi_x*phi_x)/
                pow(phi_x*phi_x+phi_y*phi_y,1.5);

            // Replace the maximum curvature
            if (abs(curvature[i+j*nx]) > maxCurvature){maxCurvature = abs(curvature[i+j*nx]);}

        }
        else {// Default value if the cell is not closed to the interface 
            curvature[i+j*nx] = 0.0;
        }
    
    }

    // Print the maximum interface curvature 
    cout << "The maximum curvature is " << maxCurvature << " m^{-2}\n";

}