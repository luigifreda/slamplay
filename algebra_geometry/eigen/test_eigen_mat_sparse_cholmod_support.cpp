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

#define EIGEN_MPL2_ONLY // this is for being sure of picking just the MPL2 licensed parts

#include <Eigen/Dense>
#include <Eigen/CholmodSupport>
#include <Eigen/SparseCore>

using namespace std;

typedef Eigen::Triplet<double> T; // A triplet is a simple object representing a non-zero entry as the triplet: row index, column index, value.

void insertCoefficientLaplaceEquation(int id, int i, int j, double w, std::vector<T>& coeffs,
                       Eigen::VectorXd& b, const Eigen::VectorXd& boundary)
{
    int n = boundary.size();
    int id1 = i + j*n;
    if (i == -1 || i == n) b(id) -= w * boundary(j); // constrained coefficient
    else if (j == -1 || j == n) b(id) -= w * boundary(i); // constrained coefficient
    else coeffs.push_back(T(id, id1, w)); // unknown coefficient
}

void buildProblemLaplaceEquation(std::vector<T>& coefficients, Eigen::VectorXd& b, int n)
{
    b.setZero();
    Eigen::ArrayXd boundary = Eigen::ArrayXd::LinSpaced(n, 0, M_PI).sin().pow(2);
    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < n; ++i)
        {
            int id = i + j*n;
            insertCoefficientLaplaceEquation(id, i - 1, j, -1, coefficients, b, boundary);
            insertCoefficientLaplaceEquation(id, i + 1, j, -1, coefficients, b, boundary);
            insertCoefficientLaplaceEquation(id, i, j - 1, -1, coefficients, b, boundary);
            insertCoefficientLaplaceEquation(id, i, j + 1, -1, coefficients, b, boundary);
            insertCoefficientLaplaceEquation(id, i, j, 4, coefficients, b, boundary);
        }
    }
}

//In the main function, we declare a list coefficients of triplets (as a std vector) 
//and the right hand side vector $ b $ which are filled by the buildProblem function. 
//The raw and flat list of non-zero entries is then converted to a true SparseMatrix object A. 
//Note that the elements of the list do not have to be sorted, and possible duplicate entries will be summed up.
void buildProblem(std::vector<T>& coefficients, Eigen::VectorXd& b, int n)
{
    buildProblemLaplaceEquation(coefficients, b, n);
}

int main()
{
    int n = 300; // size of the image
    int m = n*n; // number of unknows (=number of pixels)

    // Assembly:
    std::vector<T> coefficients; // list of non-zeros coefficients
    Eigen::VectorXd b(m); // the right hand side-vector resulting from the constraints
    buildProblem(coefficients, b, n); // prepare the coefficients as a list of triplets 

    cout << "list of coefficients set" << endl; 
    
    // ...
    Eigen::SparseMatrix<double> A(m, m); // declares a column-major sparse matrix type of double
    // fill A
    A.setFromTriplets(coefficients.begin(), coefficients.end());
    
    cout << "sparse mat set" << endl; 

    Eigen::VectorXd x;

    // solve Ax = b
    Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double> > solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success)
    {
        cout << "decomposition failed " << endl; 
        // decomposition failed
        return -1;
    }
    x = solver.solve(b);
    if (solver.info() != Eigen::Success)
    {
        cout << "solving failed " << endl; 
        // solving failed
        return -1;
    }
    cout << "solving completed" << endl;
}