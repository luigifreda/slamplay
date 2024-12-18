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
    // fixed size
    typedef Matrix<float, 4, 4> Matrix4f;
    typedef Matrix<int, 1, 2> RowVector2i;

    // dynamic size
    typedef Matrix<double, Dynamic, Dynamic> MatrixXd;
    typedef Matrix<int, Dynamic, 1> VectorXi; // column vector

    // The matrix is stored column-major

    // constructors
    Matrix3f a;
    MatrixXf b;
    MatrixXf c(10,15);
    VectorXf d(30);

    // constructors to initialize value of small fixed-size vectors
    Vector4d e(5.0, 6.0, 7.0, 8.0);

    {
    // coefficient accesors
    MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    }

    VectorXd v(2);
    v(0) = 4;
    v(1) = v(0) - 1;

    {
    // comma-initialization
    Matrix3f m;
    m << 1, 2, 3,
        4, 5, 6,
        7, 8, 9;
    std::cout << m;
    /* prints:
    1 2 3
    4 5 6
    7 8 9
    */
    }

    {
    // resizing
    MatrixXd m(2,5);
    m.resize(4,3);
    std::cout << "The matrix m is of size " << m.rows() << "x" << m.cols() << std::endl;
    std::cout << "It has " << m.size() << " coefficients" << std::endl;
    /* prints:
    The matrix m is of size 4x3
    It has 12 coefficients
    */
    }

    {
    // matrix addition, subtraction, and multiplication
    Matrix2d a;
    a << 1, 2,
        3, 4;
    MatrixXd b(2,2);
    b << 2, 3,
        1, 4;
    a + b;
    a - b;
    a += b;
    a * b;
    a = a * a; // no aliasing issue
    // scalar multiplication
    a * 2.5;   // a + 2.5 will fail

    // transpose
    a.transpose();
    a.transposeInPlace();  // a = a.transpose() would give aliasing issue

    // arithmetic reduction operations
    a.sum();		// 10
    a.prod();		// 24
    a.mean();		// 2.5
    a.minCoeff();	// 1
    a.maxCoeff();	// 4
    a.trace();		// 5
    std::ptrdiff_t i, j;
    a.minCoeff(&i, &j); // 0, 0
    }

    {
    // vector dot product and cross product
    Vector3d v(1,2,3);
    Vector3d w(0,1,2);
    v.dot(w);
    v.cross(w);
    }

    {
    // coefficient-wize operation
    // array is like matrix, but is for coefficient-wize operations
    MatrixXf m(2,2);
    MatrixXf n(2,2);
    m << 1,2,
        3,4;
    n << 5,6,
        7,8;
    m.array() * n.array();
    /* result:
    5  12
    21 32
    */
    m.cwiseProduct(n);
    /* result:
    5  12
    21 32
    */
    m.array() + 4;
    /* result:
    5 6
    7 8
    */
    }

    {
    // Interfacing with raw buffers: the Map class
    float pf[] = {1,2,3,4};
    int rows = 2; 
    int cols = 2; 
    Map<MatrixXf> mf(pf,rows,cols);  
    Map<const Vector4f> vf(pf);
    }

    {
    int array[8];
    for(int i = 0; i < 8; ++i) array[i] = i;
    cout << "Column-major:\n" << Map<Matrix<int,2,4> >(array) << endl;
    cout << "Row-major:\n" << Map<Matrix<int,2,4,RowMajor> >(array) << endl;
    cout << "Row-major using stride:\n" << Map<Matrix<int,2,4>, Unaligned, Stride<1,4> >(array) << endl;
    /* prints:
    Column-major:
    0 2 4 6
    1 3 5 7
    Row-major:
    0 1 2 3
    4 5 6 7
    Row-major using stride:
    0 1 2 3
    4 5 6 7
    */
    }

    // Reductions, visitors and broadcasting
    // https://eigen.tuxfamily.org/dox/group__TutorialReductionsVisitorsBroadcasting.html

  return 0;
}