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
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;
using Eigen::MatrixXd;


// Size set at compile time
int mainCompileTime()
{
    Matrix3d m = Matrix3d::Random();
    m = (m + Matrix3d::Constant(1.2)) * 50; //The function call Matrix3d::Constant(1.2) returns a 3-by-3 matrix expression having all coefficients equal to 1.2.
    cout << "m =" << endl << m << endl;
    Vector3d v(1,2,3);
    cout << "m * v =" << endl << m * v << endl;
    return 1;
}

// Size set at run time
int mainRunTime()
{
    MatrixXd m(2,2);  // This represents a matrix of arbitrary size (hence the X in MatrixXd), 
                      // in which every entry is a double (hence the d in MatrixXd)
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;
    return 1; 
}

// Size set at run time
int mainRunTime2()
{
    MatrixXd m = MatrixXd::Random(3,3);
    m = (m + MatrixXd::Constant(3,3,1.2)) * 50;  // The function call MatrixXd::Constant(3,3,1.2) returns a 3-by-3 matrix expression having all coefficients equal to 1.2.
    cout << "m =" << endl << m << endl;
    VectorXd v(3);
    v << 1, 2, 3;
    cout << "m * v =" << endl << m * v << endl;
    return 1;
}

// Map a buffer into a matrix 
int mainMapBuffer()
{
    double data33A[] = { 1, 2, 3,  /* row order */
                         4, 5, 6,
                         7, 8, 9};
    Map<Matrix<double,3,3,RowMajor> > map(data33A);
    cout << "map: " << map << endl; 
    map.col(1);
    return 1;
}


// block operations 
int mainBlock()
{
    Eigen::MatrixXf m(4,4);
    m <<  1, 2, 3, 4,
        5, 6, 7, 8,
        9,10,11,12,
        13,14,15,16;
    cout << "Block in the middle" << endl;
    cout << m.block<2,2>(1,1) << endl << endl;
    for (int i = 1; i <= 3; ++i)
    {
        cout << "Block of size " << i << "x" << i << endl;
        cout << m.block(0,0,i,i) << endl << endl;
    }
    return 1; 
}


int mainBlock2()
{
    Eigen::Array22f m;
    m << 1,2,
        3,4;
    Eigen::Array44f a = Eigen::Array44f::Constant(0.6);
    std::cout << "Here is the array a:\n" << a << "\n\n";
    a.block<2,2>(1,1) = m;
    std::cout << "Here is now a with m copied into its central 2x2 block:\n" << a << "\n\n";
    a.block(0,0,2,3) = a.block(2,1,2,3);
    std::cout << "Here is now a with bottom-right 2x3 block copied into top-left 2x3 block:\n" << a << "\n\n";
    return 1; 
}


int main()
{
    mainCompileTime(); 
    mainRunTime();    
    mainRunTime2();

    mainMapBuffer();
    
    mainBlock();
    mainBlock2();
    return 1; 
}