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
 This intro demonstrates the use of Eigen primitive types
 Further details here: 
 https://eigen.tuxfamily.org/dox/group__DenseMatrixManipulation__chapter.html 
************************************************************/

int main(int argc, char **argv) 
{
    // All vectors and matrices in Eigen are Eigen::Matrix, which is a template class. 
    // Its first three parameters are: data type, rows, columns. That is:
    //    Matrix<typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
    // Note that rows and columns can be:
    // - numbers   => the matrix is statically allocated 
    // - "Dynamic" => this value means a positive quantity (e.g., a size) that is not known at compile-time, and that instead is value stored in some runtime variable.
    // For instance, declare a 2*3 float matrix
    Matrix<float, 2, 3> matrix_23f;

    // At the same time, Eigen provides many built-in types through typedef, 
    // but the bottom layer is still Eigen::Matrix
    // For example, Vector3d is essentially Eigen::Matrix<double, 3, 1>, which is a three-dimensional vector
    Vector3d v_3d;   // the "d" in Vector3d stands for "double", you can replace it with f for storing instead with "float"
    //this is the same but in float format 
    Matrix<float, 3, 1> v_3f;

    //Matrix3d ​​is essentially Eigen::Matrix<double, 3, 3>
    Matrix3d matrix_33d = Matrix3d::Zero(); // initialize to zero

    // Eigen make available Matrices and Vectors up to size 4 
    Matrix4d matrix_44d = Matrix4d::Zero(); // initialize to zero
    Vector4d v_4d(0.0,0.0,0.0,0.0);
    // Matrix5d matrix_55d = Matrix5d::Zero(); // initialize to zero    // this will result in an error 

    // If you are not sure about the matrix size, you can use a dynamically sized matrix
    Matrix<double, Dynamic, Dynamic> matrix_dynamic;
    // Same thing but simpler
    MatrixXd matrix_x;
    VectorXd v_x; 
    // There are many more of this type, we will not list them one by one

    // The following is the operation on the Eigen matrix
    // input data (initialization)
    matrix_23f << 1, 2, 3, 4, 5, 6;
    //output
    cout << "matrix 2x3 from 1 to 6: \n" << matrix_23f << endl;

    //Use () to access elements in the matrix
    cout << "print matrix 2x3: " << endl;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) cout << matrix_23f(i, j) << "\t";
      cout << endl;
    }

    // Get matrix infos 
    std::cout << "rows: " << matrix_23f.rows() << std::endl; 
    std::cout << "cols: " << matrix_23f.cols() << std::endl; 


    // Matrix and vector multiplication (actually still matrix and matrix)
    v_3d << 3, 2, 1;
    v_3f << 4, 5, 6;


    // Other types of initialization 
    // 1. Interface with a data buffer 
    double data33A[] = { 1, 2, 3,  /* row order */
                         4, 5, 6,
                         7, 8, 9};
    Map<Matrix<double,3,3,RowMajor> > map(data33A);
    cout << "map: " << map << endl; 

    // In Eigen, you can't mix two different types of matrices, it's wrong like this
    // Matrix<double, 2, 1> result_wrong_type = matrix_23f *v_3d;
    // should be explicitly converted
    Matrix<double, 2, 1> result = matrix_23f.cast<double>() * v_3d;
    cout << "[1,2,3;4,5,6]*[3,2,1]=" << result.transpose() << endl;

    Matrix<float, 2, 1> result2 = matrix_23f * v_3f;
    cout << "[1,2,3;4,5,6]*[4,5,6]: " << result2.transpose() << endl;

    // Also you can't get the dimensions of the matrix wrong
    // Try to uncomment the following to see what error Eigen will report
    // Eigen::Matrix<double, 2, 3> result_wrong_dimension = matrix_23f.cast<double>() *v_3d;

    cout << endl; 

    // Some matrix operations
    // The four arithmetic operations will not be demonstrated, just use +-*/directly. See https://eigen.tuxfamily.org/dox/group__TutorialMatrixArithmetic.html
    matrix_33d = Matrix3d::Random(); // matrix of random numbers
    cout << "random matrix: \n" << matrix_33d << endl;
    cout << "transpose: \n" << matrix_33d.transpose() << endl; // Transpose
    cout << "sum: " << matrix_33d.sum() << endl; // Sum together all the elements
    cout << "trace: " << matrix_33d.trace() << endl; // trace
    cout << "times 10: \n" << 10 * matrix_33d << endl; // number multiplication
    cout << "inverse: \n" << matrix_33d.inverse() << endl; // inverse 
    cout << "det: " << matrix_33d.determinant() << endl; // determinant
    cout << "adjoint: " << matrix_33d.adjoint() << endl; // determinant  


    // Accessing cols and rows 
    std::cout << "row 1: \n" << matrix_33d.row(1) << std::endl;     
    std::cout << "col 1: \n" << matrix_33d.col(1) << std::endl; 


    // Accessing blocks. 
    // Block of size (p,q), starting at (i,j). Two versions: 
    // fixed-size block expression    =>   matrix.block<p,q>(i,j);  (compile-time)    
    // dynamic-size block expression	=>   matrix.block(i,j,p,q);   (run-time)
    Eigen::MatrixXf m(4,4);
    m <<  1, 2, 3, 4,
          5, 6, 7, 8,
          9,10,11,12,
          13,14,15,16;
    cout << "Block in the middle" << endl;
    cout << m.block<2,2>(1,1) << endl << endl; // compile-time: 2x2 block starting at (1,1)
    for (int i = 1; i <= 3; ++i)
    {
      cout << "Block of size " << i << "x" << i << endl;
      cout << m.block(0,0,i,i) << endl << endl; // run-time: ixi block starting at (0,0)
    }

    // Block for vectors. 
    // Block containing the first n=3 elements *	
    VectorXd vector;
    vector << 1,2,3,4,5,6,7,8,9,10; 
    vector.head(3);    // dynamic-size (run-time)
    vector.head<3>();  // fixed-size (compile-time )
    // Block containing the last n=3 elements *	
    vector.tail(3);
    vector.tail<3>();
    // Block containing n=3 elements, starting at position i=1 *	
    vector.segment(1,3);
    vector.segment<3>(1);

    return 0;
}