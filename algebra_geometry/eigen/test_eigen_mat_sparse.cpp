#include <iostream>

#define EIGEN_MPL2_ONLY // this is for being sure of picking just the MPL2 licensed parts

#include <Eigen/Dense>
#include <Eigen/CholmodSupport>
#include <Eigen/SparseCore>

using namespace Eigen;

int main()
{
    // < The SparseMatrix and SparseVector classes take three template arguments: the scalar type (e.g., double) 
    // < the storage order (ColMajor or RowMajor, the default is ColMajor) the inner index type (default is int).
    
    SparseMatrix<std::complex<float> > matc(1000,2000); // declares a 1000x2000 column-major compressed sparse matrix of complex<float>
    SparseMatrix<double,RowMajor> mat(1000,2000); // declares a 1000x2000 row-major compressed sparse matrix of double
    SparseVector<std::complex<float> > vecc(1000); // declares a column sparse vector of complex<float> of size 1000
    SparseVector<double,RowMajor> vec(1000); // declares a row sparse vector of double of size 1000

    // < The dimensions of a matrix can be queried using the following functions:
    //Standard dimensions	
    mat.rows();
    mat.cols();
    vec.size();
    
    //Sizes along the inner/outer dimensions	
    mat.innerSize();
    mat.outerSize(),

    //Number of non zero coefficients
    mat.nonZeros();
    vec.nonZeros();
    
    // < Iterating over the nonzero coefficients
    //Random access to the elements of a sparse object can be done through the coeffRef(i,j) function. 
    //However, this function involves a quite expensive binary search. 
    //In most cases, one only wants to iterate over the non-zeros elements. 
    //This is achieved by a standard loop over the outer dimension, 
    //and then by iterating over the non-zeros of the current inner vector via an InnerIterator. 
    //Thus, the non-zero entries have to be visited in the same order than the storage order. Here is an example: 

    {
    int rows = 100;
    int cols = rows;
    SparseMatrix<double> mat(rows,cols);
    for (int k=0; k<mat.outerSize(); ++k)
    for (SparseMatrix<double>::InnerIterator it(mat,k); it; ++it)
    {
        it.value(); // < For a writable expression, the referenced value can be modified using the valueRef() function.
        it.row(); // row index
        it.col(); // col index (here it is equal to k)
        it.index(); // inner index, here it is equal to it.row()
    }
    }
    
    {
    int size = 100;
    SparseVector<double> vec(size);
    for (SparseVector<double>::InnerIterator it(vec); it; ++it)
    {
        it.value(); // == vec[ it.index() ] // < For a writable expression, the referenced value can be modified using the valueRef() function.
        it.index();
    }
    }
    
    // < The simplest way to create a sparse matrix while guaranteeing good performance 
    // < is thus to first build a list of so-called triplets, and then convert it to a SparseMatrix.
    {
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    int estimation_of_entries = 20;
    tripletList.reserve(estimation_of_entries);
    int i = 0,j = 0;
    double v_ij; 
    // < The std::vector of triplets might contain the elements in arbitrary order, 
    // < and might even contain duplicated elements that will be summed up by setFromTriplets()
    for(int h=0;h<estimation_of_entries;h++)
    {
        // ...
        i = (i + 1) % estimation_of_entries;
        j = (j + 1) % estimation_of_entries;
        tripletList.push_back(T(i,j,v_ij));
    }
    int rows = 100;
    int cols = rows;
    SparseMatrix<double> mat(rows,cols);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    // mat is ready to go!
    }
    
    // < In some cases, however, slightly higher performance, and lower memory consumption can be reached by directly 
    // < inserting the non-zeros into the destination matrix. A typical scenario of this approach is illustrated bellow:
    {
    int rows = 100;
    int cols = rows;
    int i = 0,j = 0;
    double v_ij = 1.; 
    /*1:*/ SparseMatrix<double> mat(rows,cols); // default is column major
    /*2:*/ mat.reserve(VectorXi::Constant(cols,6)); // The key ingredient here is the line 2 where we reserve room for 6 non-zeros per column
    /*3:*/ // < for each i,j such that v_ij != 0
    /*4:*/ mat.insert(i,j) = v_ij; // alternative: mat.coeffRef(i,j) += v_ij;
    /*5:*/ mat.makeCompressed(); // optional
    }
    
    
    // < test block access of eigen sparse matrix ?? 
    
    return 1; 
}