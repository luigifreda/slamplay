// *************************************************************************
/* 
 * This file is part of the slamplay project.
 * Copyright (C) 2018-present Luigi Freda <luigifreda at gmail dot com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version, at your option. If this file is a modified/adapted 
 * version of an original file distributed under a different license that 
 * is not compatible with the GNU General Public License, the 
 * BSD 3-Clause License will apply instead.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 */
// *************************************************************************
#include <iostream>

using namespace std;

#include <ctime>
//Eigen core part
#include <Eigen/Core>
//Algebraic operations on dense matrices (inverse, eigenvalues, etc.)
#include <Eigen/Dense>

using namespace Eigen;

/* *********************************************************
 This intro demonstrates how to decompose a matrix.

 https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html

 Table of decompisitions: 
 https://eigen.tuxfamily.org/dox/group__TopicLinearAlgebraDecompositions.html
************************************************************/

int main(int argc, char **argv) 
{
    constexpr size_t mat_size = 10; 

    using MatrixNNd = Matrix<double, mat_size, mat_size>;
    using VectorNd = Matrix<double, mat_size, 1>;

    MatrixNNd mNNd = MatrixXd::Random(mat_size, mat_size);
    mNNd = mNNd * mNNd.transpose(); // Guaranteed to be positive semidefinite (real symmetric matrix can guarantee successful diagonalization)

    VectorNd vNd = MatrixXd::Random(mat_size, 1);

    // Eigenvalues  
    // Real symmetric matrix can guarantee successful diagonalization
    SelfAdjointEigenSolver<MatrixNNd> eigen_solver(mNNd);
    if (eigen_solver.info() != Eigen::Success) abort();
    VectorNd lamdas = eigen_solver.eigenvalues();
    cout << "Eigen values = \n" << lamdas << endl;
    MatrixNNd U = eigen_solver.eigenvectors();
    cout << "Eigen vectors = \n" << U << endl;    
    VectorNd singularValuesViaDiagonalization = eigen_solver.eigenvalues().cwiseSqrt();
    MatrixNNd V = U;  // In SVD, U and V are the same for symmetric matrices
    // U, V, and singularValues now contain the SVD components of A


    // SVD 
    JacobiSVD<MatrixNNd> svd(mNNd, ComputeThinU | ComputeThinV);
    VectorNd singularValues = svd.singularValues();
    cout << "SVD: singular values = \n" << singularValues << endl;         
    MatrixNNd Us = svd.matrixU();
    cout << "SVD: Us = \n" << Us << endl;       
    MatrixNNd Vs = svd.matrixV();
    cout << "SVD: Vs = \n" << Vs << endl;        

    return 0;
}