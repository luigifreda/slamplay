#pragma once

#include <iostream>
#include <fstream>  
#include <assert.h>
#include <Eigen/SparseCore>
#include "LogColors.h"

#include <limits>


//#define SPARSE_UTL_DEBUG


#ifdef SPARSE_UTL_DEBUG
using namespace std;
#endif


///	\namespace sparsematutils
///	\brief some utilities for sparse matrix management gathered in a class
///	\author 
///	\note
namespace sparsematutils
{

#ifdef SPARSE_UTL_DEBUG
    // print a list (value, index) from the inner vectors, the elements corresponding to the range [currOuterIndex,nextOuterIndex] are printed green
    inline void printInnerVectors(Eigen::SparseMatrix<double,Eigen::ColMajor>& m, int currOuterIndex, int nextOuterIndex)
    {
        if(m.isCompressed())
        {
          for (int i=0; i<m.nonZeros(); ++i) // no free spaces
          {
            if(m.outerIndexPtr()[i]==currOuterIndex) cout << LOG_COL_GREEN;
            if(m.outerIndexPtr()[i]==nextOuterIndex) cout << LOG_COL_NORMAL;
            cout << "(" << m.valuePtr()[i] << "," << m.innerIndexPtr()[i] << ") ";
          }
        }
        else
        {
          for (int i=0; i<m.outerSize(); ++i) // free spaces after the regularly filled elements 
          {
            if(m.outerIndexPtr()[i]==currOuterIndex) cout << LOG_COL_GREEN;
            if(m.outerIndexPtr()[i]==nextOuterIndex) cout << LOG_COL_NORMAL;
            int p = m.outerIndexPtr()[i];
            int pe = m.outerIndexPtr()[i]+m.innerNonZeroPtr()[i];
            int k=p;
            for (; k<pe; ++k)
              cout << "(" << m.valuePtr()[k] << "," << m.innerIndexPtr()[k] << ") ";
            for (; k<m.outerIndexPtr()[i+1]; ++k) // free spaces 
              cout << "(_,_) ";
          }
        }
        cout << LOG_COL_NORMAL << endl; 
    }
#endif
    
//    class ColWrapper
//    {
//    public:
//        // access r-th element of the column
//        virtual double operator()(int r) = 0;
//    protected:
//        ColWrapper(double* data):_pData(data){}
//                
//        double* _pData; 
//    };
//    
//    class ColWrapperRowMajor: public ColWrapper
//    {
//    public:
//        ColWrapperRowMajor(double* data, int j, int rows, int cols):ColWrapper(data),_iJ(j),_iCols(cols){}
//        double operator()(int r) { return _pData[_iJ + r*_iCols];} // nothing can be optmized 
//        
//    protected:
//        int& _iJ;
//        int& _iCols; 
//    };
//    
//    class ColWrapperColMajor: public ColWrapper
//    {
//    public:
//        ColWrapperColMajor(double* data, int j, int rows, int cols):ColWrapper(data + j*rows){}
//        double operator()(int r) { return _pData[r];}   
//    };


    // < Fill a block of a sparse matrix
    // < Sparse Matrix must be  ColMajor
    // < Input Data block must be RowMajor
    inline void setBlock(Eigen::SparseMatrix<double,Eigen::ColMajor>& M, int ibegin, int jbegin, int icount, int jcount, const double* dataRowMajor /* row major order */)
    {
#ifdef SPARSE_UTL_DEBUG
        cout << LOG_COL_CYAN << "fillBlock " << "(" << ibegin << ", " << jbegin <<", " << icount <<", "<< jcount <<")" <<   LOG_COL_NORMAL << endl; 
#endif
        assert(ibegin+icount <= M.rows());
        assert(jbegin+jcount <= M.cols());

        int mj,mi,i,j,currOuterIndex,nextOuterIndex,endOuterIndex;

        for(j=0; j<jcount; j++)
        {
            mj=j+jbegin; // this is the current working column 
#ifdef SPARSE_UTL_DEBUG
            cout << LOG_COL_GREEN << "iteration on col " << mj << LOG_COL_NORMAL << endl; 
#endif

            currOuterIndex = M.outerIndexPtr()[mj];   // index of (first element of column mj)
            nextOuterIndex = M.outerIndexPtr()[mj+1]; // index of (first element of column mj+1)

            int numColNonZeros; // number of non-zero elements in this column 
            if(M.isCompressed()) // check if mat is compressed 
            {
                numColNonZeros = nextOuterIndex - currOuterIndex; // when mat is compressed innerNonZeroPtr() is deallocated (hence useless) 
            }
            else
            {
                numColNonZeros = M.innerNonZeroPtr()[mj]; // array is redundant with empty spaces, must use innerNonZeroPtr()
            }
            endOuterIndex = currOuterIndex + numColNonZeros; // index of (last regular element of column mj), in [endOuterIndex, nextOuterIndex] we have possibly free spaces 
#ifdef SPARSE_UTL_DEBUG
            cout << "currOuterIndex: " << currOuterIndex <<", endOuterIndex: " << endOuterIndex << ", nextOuterIndex: " << nextOuterIndex << endl; 
            printInnerVectors(M,currOuterIndex,nextOuterIndex);
#endif
            // check if column exists
            if(numColNonZeros == 0)
            {
                // column does not exist
#ifdef SPARSE_UTL_DEBUG                
                cout << "column " << mj << " does not exist" << endl; 
#endif
                // insert new elements after having filled all free spaces 
                for(int nn = 0; nn<icount; nn++)
                {
#ifdef SPARSE_UTL_DEBUG   
                    cout << "inserting (" << nn+ibegin <<","<<mj<<") : " << dataRowMajor[j + nn*jcount] << endl;  
#endif
                    M.insert(nn+ibegin,mj) = dataRowMajor[j + nn*jcount];
                }
                continue; // go next column, we have done here 
            }

            i = 0; // this also counts the number of written elements 

            // check regularly stored row indexes in the range [currOuterIndex,endOuterIndex]
            // after this {} we will check in the range [endOuterIndex, nextOuterIndex]
            for(int a = currOuterIndex; (a<endOuterIndex && i<icount); a++)
            {
                mi=M.innerIndexPtr()[a]; // get row index available
#ifdef SPARSE_UTL_DEBUG   
                cout << " iteration on row " << mi << endl; 
#endif

                if(mi < ibegin) continue; // we are before our block
                if(mi >= ibegin + icount) break; // we are after our block

                // mi is now a row index in the row-range of our block 

                int miRel = mi-ibegin; // row index relative to the block of interest
#ifdef SPARSE_UTL_DEBUG  
                cout << " miRel: " << miRel << ", i:" << i << endl; 
#endif
                if(i<miRel)  // we do not have i-th element, we have to insert new elements before miRel
                {
                    for(;i < miRel;i++)  // elements [i,miRel) do not exist, insert new elements
                    {
#ifdef SPARSE_UTL_DEBUG  
                        cout << "inserting (" << i+ibegin <<","<<mj<<"): " << dataRowMajor[j + i*jcount] << endl; 
#endif
                        M.insert(i+ibegin,mj) = dataRowMajor[j + i*jcount];
                    }

                    if(i==icount) break; //if we have finished in this column break this loop

                    // given the new insertions (hence memory reallocation), we have to read again the outer indexes and then restart (i left unchanged)
                    currOuterIndex = M.outerIndexPtr()[mj];
                    nextOuterIndex = M.outerIndexPtr()[mj+1];
                    if(M.isCompressed()) // check if mat is compressed 
                    {
                        endOuterIndex = nextOuterIndex; 
                    }
                    else
                    {
                        endOuterIndex = currOuterIndex + M.innerNonZeroPtr()[mj]; // array is redundant with free spaces, must use innerNonZeroPtr()
                    }
#ifdef SPARSE_UTL_DEBUG  
                    cout << "currOuterIndex: " << currOuterIndex <<", endOuterIndex: " << endOuterIndex << ", nextOuterIndex: " << nextOuterIndex << endl; 
                    printInnerVectors(M,currOuterIndex,nextOuterIndex);
#endif

                    continue; // continue using new read outer indexes 
                }
                else if(i==miRel) // we already have i-th element, update it 
                {
#ifdef SPARSE_UTL_DEBUG  
                    cout << "updating  (" << i+ibegin <<","<<mj<<") : " << dataRowMajor[j + i*jcount] << endl; 
#endif
                    // element already exist
                    M.valuePtr()[a] = dataRowMajor[j + i*jcount];
                    i++;
                }
            }

            // we have not finished, insert new elements 
            for(;i<icount /*not finished*/;i++)
            {
#ifdef SPARSE_UTL_DEBUG  
                cout << "inserting (" << i+ibegin <<","<<mj<<") : " << dataRowMajor[j + i*jcount] << endl; 
#endif
                M.insert(i+ibegin,mj) = dataRowMajor[j + i*jcount];
            }
        }
    }

    
    // < Fill a block of a sparse matrix
    // < Sparse Matrix must be ColMajor
    // < Input Data block must be ColMajor
    inline void setBlockColMajor(Eigen::SparseMatrix<double,Eigen::ColMajor>& M, int ibegin, int jbegin, int icount, int jcount, const double* dataColMajor /* col major order */)
    {
#ifdef SPARSE_UTL_DEBUG
        cout << LOG_COL_CYAN << "fillBlock " << "(" << ibegin << ", " << jbegin <<", " << icount <<", "<< jcount <<")" <<   LOG_COL_NORMAL << endl; 
#endif
        assert(ibegin+icount <= M.rows());
        assert(jbegin+jcount <= M.cols());

        int mj,mi,i,j,currOuterIndex,nextOuterIndex,endOuterIndex;

        for(j=0; j<jcount; j++)
        {
            const double* datacj = dataColMajor + j*icount; // start of column j
            
            mj=j+jbegin; // this is the current working column 
#ifdef SPARSE_UTL_DEBUG
            cout << LOG_COL_GREEN << "iteration on col " << mj << LOG_COL_NORMAL << endl; 
#endif

            currOuterIndex = M.outerIndexPtr()[mj];   // index of (first element of column mj)
            nextOuterIndex = M.outerIndexPtr()[mj+1]; // index of (first element of column mj+1)

            int numColNonZeros; // number of non-zero elements in this column 
            if(M.isCompressed()) // check if mat is compressed 
            {
                numColNonZeros = nextOuterIndex - currOuterIndex; // when mat is compressed innerNonZeroPtr() is deallocated (hence useless) 
            }
            else
            {
                numColNonZeros = M.innerNonZeroPtr()[mj]; // array is redundant with empty spaces, must use innerNonZeroPtr()
            }
            endOuterIndex = currOuterIndex + numColNonZeros; // index of (last regular element of column mj), in [endOuterIndex, nextOuterIndex] we have possibly free spaces 
#ifdef SPARSE_UTL_DEBUG
            cout << "currOuterIndex: " << currOuterIndex <<", endOuterIndex: " << endOuterIndex << ", nextOuterIndex: " << nextOuterIndex << endl; 
            printInnerVectors(M,currOuterIndex,nextOuterIndex);
#endif
            // check if column exists
            if(numColNonZeros == 0)
            {
                // column does not exist
#ifdef SPARSE_UTL_DEBUG                
                cout << "column " << mj << " does not exist" << endl; 
#endif
                // insert new elements 
                for(int nn = 0; nn<icount; nn++)
                {
#ifdef SPARSE_UTL_DEBUG   
                    cout << "inserting (" << nn+ibegin <<","<<mj<<") : " << datacj[nn] << endl;  
#endif
                    M.insert(nn+ibegin,mj) = datacj[nn];
                }

                continue; // go next column, we have done here 
            }

            i = 0; // this also counts the number of written elements 

            // check regularly stored row indexes in the range [currOuterIndex,endOuterIndex]
            // after this {} we will check in the range [endOuterIndex, nextOuterIndex]
            for(int a = currOuterIndex; (a<endOuterIndex && i<icount); a++)
            {
                mi=M.innerIndexPtr()[a]; // get row index available
#ifdef SPARSE_UTL_DEBUG   
                cout << " iteration on row " << mi << endl; 
#endif

                if(mi < ibegin) continue; // we are before our block
                if(mi >= ibegin + icount) break; // we are after our block

                // mi is now a row index in the row-range of our block 

                int miRel = mi-ibegin; // row index relative to the block of interest
#ifdef SPARSE_UTL_DEBUG  
                cout << " miRel: " << miRel << ", i:" << i << endl; 
#endif
                if(i<miRel)  // we do not have i-th element, we have to insert new elements before miRel
                {
                    for(;i < miRel;i++)  // elements [i,miRel) do not exist, insert new elements
                    {
#ifdef SPARSE_UTL_DEBUG  
                        cout << "inserting (" << i+ibegin <<","<<mj<<"): " << datacj[i] << endl; 
#endif
                        M.insert(i+ibegin,mj) = datacj[i] ;
                    }

                    if(i==icount) break; //if we have finished in this column break this loop

                    // given the new insertions (hence memory reallocation), we have to read again the outer indexes and then restart (i left unchanged)
                    currOuterIndex = M.outerIndexPtr()[mj];
                    nextOuterIndex = M.outerIndexPtr()[mj+1];
                    if(M.isCompressed()) // check if mat is compressed 
                    {
                        endOuterIndex = nextOuterIndex; 
                    }
                    else
                    {
                        endOuterIndex = currOuterIndex + M.innerNonZeroPtr()[mj]; // array is redundant with free spaces, must use innerNonZeroPtr()
                    }
#ifdef SPARSE_UTL_DEBUG  
                    cout << "currOuterIndex: " << currOuterIndex <<", endOuterIndex: " << endOuterIndex << ", nextOuterIndex: " << nextOuterIndex << endl; 
                    printInnerVectors(M,currOuterIndex,nextOuterIndex);
#endif

                    continue; // continue using new read outer indexes 
                }
                else if(i==miRel) // we already have i-th element, update it 
                {
#ifdef SPARSE_UTL_DEBUG  
                    cout << "updating  (" << i+ibegin <<","<<mj<<") : " << datacj[i]  << endl; 
#endif
                    // element already exist
                    M.valuePtr()[a] = datacj[i] ;
                    i++;
                }
            }

            // we have not finished, insert new elements 
            for(;i<icount /*not finished*/;i++)
            {
#ifdef SPARSE_UTL_DEBUG  
                cout << "inserting (" << i+ibegin <<","<<mj<<") : " << datacj[i]  << endl; 
#endif
                M.insert(i+ibegin,mj) = datacj[i] ;
            }
        }
    }
    
    
    // < Add to a block of a sparse matrix
    // < Sparse Matrix must be ColMajor
    // < Input Data block must be ColMajor
    inline void add2BlockColMajor(Eigen::SparseMatrix<double,Eigen::ColMajor>& M, int ibegin, int jbegin, int icount, int jcount, const double* dataColMajor /* col major order */)
    {
#ifdef SPARSE_UTL_DEBUG
        cout << LOG_COL_CYAN << "fillBlock " << "(" << ibegin << ", " << jbegin <<", " << icount <<", "<< jcount <<")" <<   LOG_COL_NORMAL << endl; 
#endif
        assert(ibegin+icount <= M.rows());
        assert(jbegin+jcount <= M.cols());

        int mj,mi,i,j,currOuterIndex,nextOuterIndex,endOuterIndex;

        for(j=0; j<jcount; j++)
        {
            const double* datacj = dataColMajor + j*icount; // start of column j
            
            mj=j+jbegin; // this is the current working column 
#ifdef SPARSE_UTL_DEBUG
            cout << LOG_COL_GREEN << "iteration on col " << mj << LOG_COL_NORMAL << endl; 
#endif

            currOuterIndex = M.outerIndexPtr()[mj];   // index of (first element of column mj)
            nextOuterIndex = M.outerIndexPtr()[mj+1]; // index of (first element of column mj+1)

            int numColNonZeros; // number of non-zero elements in this column 
            if(M.isCompressed()) // check if mat is compressed 
            {
                numColNonZeros = nextOuterIndex - currOuterIndex; // when mat is compressed innerNonZeroPtr() is deallocated (hence useless) 
            }
            else
            {
                numColNonZeros = M.innerNonZeroPtr()[mj]; // array is redundant with empty spaces, must use innerNonZeroPtr()
            }
            endOuterIndex = currOuterIndex + numColNonZeros; // index of (last regular element of column mj), in [endOuterIndex, nextOuterIndex] we have possibly free spaces 
#ifdef SPARSE_UTL_DEBUG
            cout << "currOuterIndex: " << currOuterIndex <<", endOuterIndex: " << endOuterIndex << ", nextOuterIndex: " << nextOuterIndex << endl; 
            printInnerVectors(M,currOuterIndex,nextOuterIndex);
#endif
            // check if column exists
            if(numColNonZeros == 0)
            {
                // column does not exist
#ifdef SPARSE_UTL_DEBUG                
                cout << "column " << mj << " does not exist" << endl; 
#endif
                // insert new elements 
                for(int nn = 0; nn<icount; nn++)
                {
#ifdef SPARSE_UTL_DEBUG   
                    cout << "inserting (" << nn+ibegin <<","<<mj<<") : " << datacj[nn] << endl;  
#endif
                    M.insert(nn+ibegin,mj) = datacj[nn];
                }

                continue; // go next column, we have done here 
            }

            i = 0; // this also counts the number of written elements 

            // check regularly stored row indexes in the range [currOuterIndex,endOuterIndex]
            for(int a = currOuterIndex; (a<endOuterIndex && i<icount); a++)
            {
                mi=M.innerIndexPtr()[a]; // get row index available
#ifdef SPARSE_UTL_DEBUG   
                cout << " iteration on row " << mi << endl; 
#endif

                if(mi < ibegin) continue; // we are before our block
                if(mi >= ibegin + icount) break; // we are after our block

                // mi is now a row index in the row-range of our block 

                int miRel = mi-ibegin; // row index relative to the block of interest
#ifdef SPARSE_UTL_DEBUG  
                cout << " miRel: " << miRel << ", i:" << i << endl; 
#endif
                if(i<miRel)  // we do not have i-th element, we have to insert new elements before miRel
                {
                    for(;i < miRel;i++)  // elements [i,miRel) do not exist, insert new elements
                    {
#ifdef SPARSE_UTL_DEBUG  
                        cout << "inserting (" << i+ibegin <<","<<mj<<"): " << datacj[i] << endl; 
#endif
                        M.insert(i+ibegin,mj) = datacj[i] ;
                    }

                    if(i==icount) break; //if we have finished in this column break this loop

                    // given the new insertions (hence memory reallocation), we have to read again the outer indexes and then restart (i left unchanged)
                    currOuterIndex = M.outerIndexPtr()[mj];
                    nextOuterIndex = M.outerIndexPtr()[mj+1];
                    if(M.isCompressed()) // check if mat is compressed 
                    {
                        endOuterIndex = nextOuterIndex; 
                    }
                    else
                    {
                        endOuterIndex = currOuterIndex + M.innerNonZeroPtr()[mj]; // array is redundant with free spaces, must use innerNonZeroPtr()
                    }
#ifdef SPARSE_UTL_DEBUG  
                    cout << "currOuterIndex: " << currOuterIndex <<", endOuterIndex: " << endOuterIndex << ", nextOuterIndex: " << nextOuterIndex << endl; 
                    printInnerVectors(M,currOuterIndex,nextOuterIndex);
#endif

                    continue; // continue using new read outer indexes 
                }
                else if(i==miRel) // we already have i-th element, update it 
                {
#ifdef SPARSE_UTL_DEBUG  
                    cout << "updating  (" << i+ibegin <<","<<mj<<") : " << datacj[i]  << endl; 
#endif
                    // element already exist
                    M.valuePtr()[a] += datacj[i] ; // < add operation 
                    i++;
                }
            }

            // we have not finished, insert new elements 
            for(;i<icount /*not finished*/;i++)
            {
#ifdef SPARSE_UTL_DEBUG  
                cout << "inserting (" << i+ibegin <<","<<mj<<") : " << datacj[i]  << endl; 
#endif
                M.insert(i+ibegin,mj) = datacj[i] ;
            }
        }
    }
    
    
    
    // < Add to a block of a sparse matrix
    // < Sparse Matrix must be ColMajor
    // < Input Data block must be ColMajor
    inline void add2BlockColMajor2(Eigen::SparseMatrix<double,Eigen::ColMajor>& M, int ibegin, int jbegin, int icount, int jcount, const double* dataColMajor /* col major order */)
    {        
        assert(ibegin+icount <= M.rows());
        assert(jbegin+jcount <= M.cols());

        for(int j=0; j<jcount; j++)
        {
            int mj=j+jbegin; // this is the current working column 
            const double* datacj = dataColMajor + j*icount; // start of column j in buffer data
            for(int i=0; i < icount; i++)
            {
                M.coeffRef(i+ibegin,mj) += datacj[i];
            }
        }
    }
    
    inline void savePatternEps(const Eigen::SparseMatrix<double,Eigen::ColMajor>& M, const std::string& filename)
    {
        int x = M.cols();
        int y = M.rows();
        // find a scale factor that yields valid EPS coordinates
        int m = std::max(x, y);
        double scale = 1.;
        for (; scale * m >= 10000.; scale *= 0.1);
        // create file
        std::ofstream out(filename.c_str());
        out << "%!PS-Adobe-3.0 EPSF-3.0\n"
                "%%BoundingBox: 0 0 " << x * scale << " " << y * scale << "\n"
                "/BP{" << scale << " " << -scale << " scale 0 " << -y << " translate}bind def\n"
                "BP\n"
                "150 dict begin\n"
                "/D/dup cvx def/S/stroke cvx def\n"
                "/L/lineto cvx def/M/moveto cvx def\n"
                "/RL/rlineto cvx def/RM/rmoveto cvx def\n"
                "/GS/gsave cvx def/GR/grestore cvx def\n"
                "/REC{M 0 1 index RL 1 index 0 RL neg 0 exch RL neg 0 RL}bind def\n"
                "0 0 150 setrgbcolor\n"
                "0.01 setlinewidth\n";
    //    for (int row = 0; row < M.rows(); row++) 
    //    {
    //        for (SparseVectorIter iter(*_rows[row]); iter.valid(); iter.next()) 
    //        {
    //            double val;
    //            int col = iter.get(val);
    //            out << "1 1 " << col << " " << row << " REC GS fill GR S" << endl;
    //        }
    //    }
        
        for (int k=0; k<M.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(M,k); it; ++it)
        {
            //it.value();
            //it.row(); // row index
            //it.col(); // col index (here it is equal to k)
            //it.index(); // inner index, here it is equal to it.row()
            if(fabs(it.value()) > std::numeric_limits<double>::epsilon())
                out << "1 1 " << it.col() << " " << it.row() << " REC GS fill GR S" << std::endl;
        }

        out.close();
    }

}
