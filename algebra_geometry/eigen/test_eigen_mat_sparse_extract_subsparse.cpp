#include <iostream>

#define EIGEN_MPL2_ONLY // this is for being sure of picking just the MPL2 licensed parts

#include <Eigen/Dense>
#include <Eigen/CholmodSupport>
#include <Eigen/SparseCore>

using namespace std;
using namespace Eigen;

// < from  http://stackoverflow.com/questions/13094372/extract-a-block-from-a-sparse-matrix-as-another-sparse-matric

typedef Triplet<double> Tri;

// < extract a block from a sparse matrix as a sparse block - only for ColMajor Sparse Matrix
SparseMatrix<double> sparseBlock(SparseMatrix<double,ColMajor>& M, int ibegin, int jbegin, int icount, int jcount)
{
    //only for ColMajor Sparse Matrix
    assert(ibegin+icount <= M.rows());
    assert(jbegin+jcount <= M.cols());
    
    int Mj,Mi,i,j,currOuterIndex,nextOuterIndex;
    vector<Tri> tripletList;
    tripletList.reserve(M.nonZeros());

    for(j=0; j<jcount; j++)
    {
        Mj=j+jbegin;
        currOuterIndex = M.outerIndexPtr()[Mj];
        nextOuterIndex = M.outerIndexPtr()[Mj+1];

        for(int a = currOuterIndex; a<nextOuterIndex; a++)
        {
            Mi=M.innerIndexPtr()[a];

            if(Mi < ibegin) continue;
            if(Mi >= ibegin + icount) break;

            i=Mi-ibegin;    
            tripletList.push_back(Tri(i,j,M.valuePtr()[a]));
        }
    }
    SparseMatrix<double> matS(icount,jcount);
    matS.setFromTriplets(tripletList.begin(), tripletList.end());
    return matS;
}

SparseMatrix<double> sparseTopLeftBlock(SparseMatrix<double>& M, int icount, int jcount)
{
    return sparseBlock(M,0,0,icount,jcount);
}

SparseMatrix<double> sparseTopRightBlock(SparseMatrix<double>& M, int icount, int jcount)
{
    return sparseBlock(M,0,M.cols()-jcount,icount,jcount);
}

SparseMatrix<double> sparseBottomLeftBlock(SparseMatrix<double>& M, int icount, int jcount)
{
    return sparseBlock(M,M.rows()-icount,0,icount,jcount);
}

SparseMatrix<double> sparseBottomRightBlock(SparseMatrix<double>& M, int icount, int jcount)
{
    return sparseBlock(M,M.rows()-icount,M.cols()-jcount,icount,jcount);
}

int main()
{
    // < The SparseMatrix and SparseVector classes take three template arguments: the scalar type (e.g., double) 
    // < the storage order (ColMajor or RowMajor, the default is ColMajor) the inner index type (default is int).
    
    SparseMatrix<double> ms(20,20); // declares a ColMajor compressed sparse matrix of double

    MatrixXd md = MatrixXd::Identity(10, 10); 
    
    
    cout << "md: " << md << endl; 
    cout << "ms: " << ms << endl;
    
    
    return 1; 
}