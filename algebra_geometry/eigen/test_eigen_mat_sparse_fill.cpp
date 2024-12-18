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

#define EIGEN_MPL2_ONLY  // this is for being sure of picking just the MPL2 licensed parts

#include <Eigen/CholmodSupport>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "sparse/SparseMatUtils.h"

#include "io/LogColors.h"

using namespace std;
using namespace Eigen;
using namespace slamplay;
using namespace sparsematutils;

//
//// print a list (value, index) from the inner vectors, the elements corresponding to the range [currOuterIndex,nextOuterIndex] are printed green
// void printInnerVectors(SparseMatrix<double,ColMajor>& m, int currOuterIndex, int nextOuterIndex)
//{
//     if(m.isCompressed())
//     {
//       for (int i=0; i<m.nonZeros(); ++i) // no free spaces
//       {
//         if(m.outerIndexPtr()[i]==currOuterIndex) cout << LOG_COL_GREEN;
//         if(m.outerIndexPtr()[i]==nextOuterIndex) cout << LOG_COL_NORMAL;
//         cout << "(" << m.valuePtr()[i] << "," << m.innerIndexPtr()[i] << ") ";
//       }
//     }
//     else
//     {
//       for (int i=0; i<m.outerSize(); ++i) // free spaces after the regularly filled elements
//       {
//         if(m.outerIndexPtr()[i]==currOuterIndex) cout << LOG_COL_GREEN;
//         if(m.outerIndexPtr()[i]==nextOuterIndex) cout << LOG_COL_NORMAL;
//         int p = m.outerIndexPtr()[i];
//         int pe = m.outerIndexPtr()[i]+m.innerNonZeroPtr()[i];
//         int k=p;
//         for (; k<pe; ++k)
//           cout << "(" << m.valuePtr()[k] << "," << m.innerIndexPtr()[k] << ") ";
//         for (; k<m.outerIndexPtr()[i+1]; ++k) // free spaces
//           cout << "(_,_) ";
//       }
//     }
//     cout << LOG_COL_NORMAL << endl;
// }
//
//// < fill a block of a sparse matrix  - only for ColMajor Sparse Matrix
// void fillBlock(SparseMatrix<double,ColMajor>& M, int ibegin, int jbegin, int icount, int jcount, double* dataRowMajor /* row major order */)
//{
//     cout << LOG_COL_CYAN << "fillBlock " << "(" << ibegin << ", " << jbegin <<", " << icount <<", "<< jcount <<")" <<   LOG_COL_NORMAL << endl;
//     assert(ibegin+icount <= M.rows());
//     assert(jbegin+jcount <= M.cols());
//
//     int mj,mi,i,j,currOuterIndex,nextOuterIndex,endOuterIndex;
//
//     for(j=0; j<jcount; j++)
//     {
//         mj=j+jbegin; // this is the current working column
//         cout << LOG_COL_GREEN << "iteration on col " << mj << LOG_COL_NORMAL << endl;
//
//         currOuterIndex = M.outerIndexPtr()[mj];   // index of (first element of column mj)
//         nextOuterIndex = M.outerIndexPtr()[mj+1]; // index of (first element of column mj+1)
//
//         int numColNonZeros; // number of non-zero elements in this column
//         if(M.isCompressed()) // check if mat is compressed
//         {
//             numColNonZeros = nextOuterIndex - currOuterIndex; // when mat is compressed innerNonZeroPtr() is deallocated (hence useless)
//         }
//         else
//         {
//             numColNonZeros = M.innerNonZeroPtr()[mj]; // array is redundant with empty spaces, must use innerNonZeroPtr()
//         }
//         endOuterIndex = currOuterIndex + numColNonZeros; // index of (last regular element of column mj), in [endOuterIndex, nextOuterIndex] we have possibly free spaces
//
//         cout << "currOuterIndex: " << currOuterIndex <<", endOuterIndex: " << endOuterIndex << ", nextOuterIndex: " << nextOuterIndex << endl;
//         printInnerVectors(M,currOuterIndex,nextOuterIndex);
//
//         // check if column exists
//         if(numColNonZeros == 0)
//         {
//             // column does not exist
//             cout << "column " << mj << " does not exist" << endl;
//
//             // use free space if available in the range [endOuterIndex, nextOuterIndex]
//             int nn = 0; // this also counts the number of free spaces we fill
//             for (int k = endOuterIndex; ( k<nextOuterIndex && nn < icount); k++, nn++)
//             {
//                 cout << "updating (freespace)  (" << nn+ibegin <<","<<mj<<") : " << dataRowMajor[j + nn*jcount] << endl;
//                 M.valuePtr()[k] = dataRowMajor[j + nn*jcount];
//                 M.innerIndexPtr()[k] = nn+ibegin;
//                 M.innerNonZeroPtr()[mj]++; // this can be used since the matrix is non compressed (endOuterIndex is smaller than nextOuterIndex)
//             }
//
//             // if needed, insert new elements after having filled all free spaces
//             for(; nn<icount; nn++)
//             {
//                 cout << "inserting (" << nn+ibegin <<","<<mj<<") : " << dataRowMajor[j + nn*jcount] << endl;
//                 M.insert(nn+ibegin,mj) = dataRowMajor[j + nn*jcount];
//             }
//
//             continue; // go next column, we have done here
//         }
//
//         i = 0; // this also counts the number of written elements
//
//         // check regularly stored row indexes in the range [currOuterIndex,endOuterIndex]
//         // after this {} we will check in the range [endOuterIndex, nextOuterIndex]
//         for(int a = currOuterIndex; (a<endOuterIndex && i<icount); a++)
//         {
//             mi=M.innerIndexPtr()[a]; // get row index available
//             cout << " iteration on row " << mi << endl;
//
//             if(mi < ibegin) continue; // we are before our block
//             if(mi >= ibegin + icount) break; // we are after our block
//
//             // mi is now a row index in the row-range of our block
//
//             int miRel = mi-ibegin; // row index relative to the block of interest
//             cout << " miRel: " << miRel << ", i:" << i << endl;
//             if(i<miRel)  // we do not have i-th element, we have to insert new elements before miRel
//             {
//                 for(;i < miRel;i++)  // elements [i,miRel) do not exist, insert new elements
//                 {
//                     cout << "inserting (" << i+ibegin <<","<<mj<<"): " << dataRowMajor[j + i*jcount] << endl;
//                     M.insert(i+ibegin,mj) = dataRowMajor[j + i*jcount];
//                 }
//
//                 if(i==icount) break; //if we have finished in this column break this loop
//
//                 // given the new insertions (hence memory reallocation), we have to read again the outer indexes and then restart (i left unchanged)
//                 currOuterIndex = M.outerIndexPtr()[mj];
//                 nextOuterIndex = M.outerIndexPtr()[mj+1];
//                 if(M.isCompressed()) // check if mat is compressed
//                 {
//                     endOuterIndex = nextOuterIndex;
//                 }
//                 else
//                 {
//                     endOuterIndex = currOuterIndex + M.innerNonZeroPtr()[mj]; // array is redundant with free spaces, must use innerNonZeroPtr()
//                 }
//                 cout << "currOuterIndex: " << currOuterIndex <<", endOuterIndex: " << endOuterIndex << ", nextOuterIndex: " << nextOuterIndex << endl;
//                 printInnerVectors(M,currOuterIndex,nextOuterIndex);
//
//                 continue; // continue using new read outer indexes
//             }
//             else if(i==miRel) // we already have i-th element, update it
//             {
//                 cout << "updating  (" << i+ibegin <<","<<mj<<") : " << dataRowMajor[j + i*jcount] << endl;
//                 // element already exist
//                 M.valuePtr()[a] = dataRowMajor[j + i*jcount];
//                 i++;
//             }
//         }
//
//         // if not finished, use free spaces if available in the range [endOuterIndex, nextOuterIndex]
//         for(int a=endOuterIndex; ( a<nextOuterIndex && i<icount/*not finished*/); a++)
//         {
//             cout << "updating (freespace)  (" << i+ibegin <<","<<mj<<") : " << dataRowMajor[j + i*jcount] << endl;
//             // element already exist
//             M.valuePtr()[a] = dataRowMajor[j + i*jcount];
//             M.innerIndexPtr()[a] = i+ibegin;
//             M.innerNonZeroPtr()[mj]++; // this can be used since the matrix is non compressed (endOuterIndex is smaller than nextOuterIndex)
//             i++;
//         }
//
//         // we have not finished by using free spaces, insert new elements
//         for(;i<icount /*not finished*/;i++)
//         {
//             cout << "inserting (" << i+ibegin <<","<<mj<<") : " << dataRowMajor[j + i*jcount] << endl;
//             M.insert(i+ibegin,mj) = dataRowMajor[j + i*jcount];
//         }
//     }
// }

// < multiply diagonal element per lambda - only for ColMajor Sparse Matrix
// void multDiagonal(SparseMatrix<double,ColMajor>& m, double lambda)
//{
//    for(int j=0; j<m.cols(); j++)
//    {
//        m.coeffRef(j,j) *= lambda;
//    }
//}

int main() {
    // < The SparseMatrix and SparseVector classes take three template arguments: the scalar type (e.g., double)
    // < the storage order (ColMajor or RowMajor, the default is ColMajor) the inner index type (default is int).

    SparseMatrix<double> ms(10, 10);  // declares a ColMajor compressed sparse matrix of double

    // MatrixXd md = MatrixXd::Identity(10, 10);

    double data33A[] = {1, 2, 3, /* row order */
                        4, 5, 6,
                        7, 8, 9};

    double data33B[] = {9, 8, 7, /* row order */
                        6, 5, 4,
                        3, 2, 1};

    double data22A[] = {1, 3, /* row order */
                        5, 7};

    double data22B[] = {2, 4, /* row order */
                        6, 8};

    double data55A[] = {9, 8, 7, 6, 5, /* row order */
                        4, 3, 2, 1, 9,
                        9, 8, 7, 6, 5,
                        4, 3, 2, 1, 9,
                        9, 8, 7, 6, 5};

    setBlock(ms, 2, 2, 3, 3, data33A);
    cout << "ms: \n"
         << ms << endl;

    setBlock(ms, 2, 2, 3, 3, data33B);
    cout << "ms: \n"
         << ms << endl;

    setBlock(ms, 4, 4, 2, 2, data22A);
    cout << "ms: \n"
         << ms << endl;

    setBlock(ms, 8, 8, 2, 2, data22A);
    cout << "ms: \n"
         << ms << endl;

    // ms.makeCompressed();

    setBlock(ms, 5, 5, 3, 3, data33B);
    cout << "ms: " << ms << endl;

    //    fillBlock(ms, 0, 0, 5, 5, data55A );
    //    cout << "ms: " << ms << endl;

    setBlock(ms, 0, 0, 3, 3, data33A);
    cout << "ms: \n"
         << ms << endl;

    setBlock(ms, 6, 0, 2, 2, data22A);
    cout << "ms: \n"
         << ms << endl;

    setBlock(ms, 4, 0, 3, 3, data33A);
    cout << "ms: \n"
         << ms << endl;

    ///--------------------------------

    double data33SymA[] = {1, 8, 7, /* row order */
                           8, 2, 2,
                           7, 2, 3};

    SparseMatrix<double> ms2(10, 10);   // declares a ColMajor compressed sparse matrix of double
    SparseMatrix<double> ms2A(10, 10);  // declares a ColMajor compressed sparse matrix of double
    SparseMatrix<double> msI(10, 10);   // declares a ColMajor compressed sparse matrix of double
    msI.setIdentity();

    add2BlockColMajor(ms2, 2, 2, 3, 3, data33SymA);
    add2BlockColMajor(ms2, 2, 7, 3, 3, data33SymA);
    cout << "ms2: \n"
         << ms2 << endl;

    double lambda = 2;
    //    ms2.cwiseProduct((1+lambda)*msI); //does not work !

    ms2A = ms2;
    for (int j = 0; j < ms2A.cols(); j++)
    {
        ms2A.coeffRef(j, j) *= (1 + lambda);
    }

    cout << "ms2: \n"
         << ms2 << endl;
    cout << "ms2A: \n"
         << ms2A << endl;

    savePatternEps(ms2A, "prova.eps");

    return 1;
}