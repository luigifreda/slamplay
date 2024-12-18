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
 This intro demonstrates how to solve a system of equations.

 Plus, check the tutorial at the following link and take a look at 
 the table that explains the different options in terms of 
 accuracy-speed trade-off
 https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html

 See also this table of decompisitions: 
 https://eigen.tuxfamily.org/dox/group__TopicLinearAlgebraDecompositions.html 
************************************************************/

int main(int argc, char **argv) 
{
      // Solving equations: We solve the equation mNNd * x = vNd
      // The size of N is defined in the following macro.

      constexpr size_t mat_size = 50; 
    
      using MatrixNNd = Matrix<double, mat_size, mat_size>;
      using VectorNd = Matrix<double, mat_size, 1>;

      MatrixNNd mNNd = MatrixXd::Random(mat_size, mat_size);
      mNNd = mNNd * mNNd.transpose(); // Guaranteed to be positive semidefinite

      VectorNd vNd = MatrixXd::Random(mat_size, 1);

      clock_t time_stt = clock(); // timing
      double relative_error = 0; 


      // Direct inversion is naturally the most direct, but the amount of inversion calculation is large      
      VectorNd x = mNNd.inverse() * vNd;
      cout << "time of normal inverse is "
            << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
      cout << "x = " << x.transpose() << endl;
      relative_error = (mNNd*x - vNd).norm() / vNd.norm(); // norm() is L2 norm
      std::cout << "relative error is: " << relative_error << std::endl;  
      std::cout << endl; 


      // Usually use matrix decomposition, such as QR decomposition, the speed will be much faster
      time_stt = clock();
      x = mNNd.colPivHouseholderQr().solve(vNd);
      cout << "time of Qr decomposition is "
            << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
      cout << "x = " << x.transpose() << endl;
      relative_error = (mNNd*x - vNd).norm() / vNd.norm(); // norm() is L2 norm
      std::cout << "relative error is: " << relative_error << std::endl;
      std::cout << endl; 


      // For positive definite matrices, you can also use Cholesky decomposition to solve the equation
      time_stt = clock();
      x = mNNd.ldlt().solve(vNd);
      cout << "time of ldlt decomposition is "
            << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
      cout << "x = " << x.transpose() << endl;
      relative_error = (mNNd*x - vNd).norm() / vNd.norm(); // norm() is L2 norm
      std::cout << "relative error is: " << relative_error << std::endl;
      std::cout << endl; 


      time_stt = clock();
      x = mNNd.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(vNd);
      cout << "time of bdcSvd decomposition is "
            << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
      cout << "x = " << x.transpose() << endl;
      relative_error = (mNNd*x - vNd).norm() / vNd.norm(); // norm() is L2 norm
      std::cout << "relative error is: " << relative_error << std::endl;    
      std::cout << endl;       


      time_stt = clock();
      x = mNNd.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(vNd);
      cout << "time of jacobiSvd decomposition is "
            << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
      cout << "x = " << x.transpose() << endl;
      relative_error = (mNNd*x - vNd).norm() / vNd.norm(); // norm() is L2 norm
      std::cout << "relative error is: " << relative_error << std::endl;    
      std::cout << endl;     

      /*
      Certain decompositions are rank-revealing, i.e. are able to compute the rank of a matrix. 
      These are typically also the decompositions that behave best in the face of a non-full-rank matrix 
      (which in the square case means a singular matrix). 
      On this table you can see for all our decompositions whether they are rank-revealing or not.
      https://eigen.tuxfamily.org/dox/group__TopicLinearAlgebraDecompositions.html 
      */
      Eigen::FullPivLU<MatrixNNd> lu_decomp(mNNd);
      std::cout << "The rank of mNNd is " << lu_decomp.rank() << std::endl;
      std::cout << "Here is a matrix whose columns form a basis of the null-space of mNNd:\n"
            << lu_decomp.kernel() << std::endl;
      std::cout << "Here is a matrix whose columns form a basis of the column-space of mNNd:\n"
            << lu_decomp.image(mNNd) << std::endl; // yes, have to pass the original mNNd

      return 0;
}