#include <math.h>
#include <iostream>

#include "diagnostics.h"

using namespace std;

// The total interface length (L) is computed by following the following algorithm
// L ~ sum_{i,j} delta(phi_{i,j}) norm (grad(phi)) dx dy
// with delta(phi) an approximation of the dirac function := delta(phi) = 1 / sqrt (2 * pi * epsilon) * exp (- phi*phi / 2 / epsilon) 
void computeInterfaceLength(double** phi, const int nx, const int ny, const double dx, const double dy){

    // Fixed parameter for the dirac function
    double epsilon = 0.001;
    
    // Interface length
    double length = 0.0;

    // Loop over all of the internal cells
    for (int i = 1; i < nx - 1; i++) {
        for (int j = 1; j < ny - 1; j++) {

            // Compute gradient of phi : grad(phi)
            double phi_x = (phi[i+1][j]-phi[i-1][j])/2.0/dx; 
            double phi_y = (phi[i][j+1]-phi[i][j-1])/2.0/dy; 

            // Compute the norm of gradient: norm(grad(phi)) 
            double normGrad = sqrt(phi_x*phi_x + phi_y*phi_y);

            // Compute the dirac function approximation
            double delta = (1.0 / sqrt(2.0 * M_PI * epsilon)) * exp( - (phi[i][j] * phi[i][j]) / (2.0 * epsilon) ) ;

            // L = delta * norm(grad(phi)) * dx * dy
            length += delta * normGrad * dx * dy;
        }
    }

    // Print the total interface length
}


// The interface curvature (L) is computed by following the following algorithm
// curvature = (phi_xx * phi_y **2 - 2.0 * phi_x * phi_y * phi_xy + phi_yy * phi_x **2) / (phi_x **2 + phi_y **2) ** (3/2) 
// with phi_x:= d phi / dx, phi_y:= d phi / dy 
// and phi_xx:= d phi_x / dx, phi_yy:= d phi_y / dy, phi_xy:= d phi_x / dy 
void computeInterfaceCurvature(double** phi,double** curvature, const int nx, const int ny, const double dx, const double dy){

    double maxCurvature = 0.0;

    for (int i = 1; i < nx-1; i++){
        for (int j = 1; j < ny-1; j++){

            if (abs(phi[i][j]) < 3.0 * dx ){ //Compute the curvature only near the interface 

                // first derivative
                double phi_x = (phi[i+1][j]-phi[i-1][j])/2.0/dx;
                double phi_y = (phi[i][j+1]-phi[i][j-1])/2.0/dy;

                // second derivative
                double phi_xx = (phi[i+1][j]-2.0*phi[i][j]+phi[i-1][j])/dx/dx;
                double phi_yy = (phi[i][j+1]-2.0*phi[i][j]+phi[i][j-1])/dy/dy;
                double phi_xy = (phi[i+1][j+1] - phi[i+1][j-1] - phi[i-1][j+1] + phi[i-1][j-1])/dx/dy/4.0;

                // compute curvature
                curvature[i][j] = (phi_xx*phi_y*phi_y - 2.0* phi_x*phi_y*phi_xy + phi_yy*phi_x*phi_x)/
                    pow(phi_x*phi_x+phi_y*phi_y,1.5);

                // Replace the maximum curvature
                if (abs(curvature[i][j]) > maxCurvature){maxCurvature = abs(curvature[i][j]);}

            }
            else {// Default value if the cell is not closed to the interface 
                curvature[i][j] = 0.0;
            }
        }
    }

    // Print the maximum interface curvature
}