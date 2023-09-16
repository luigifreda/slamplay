	
#pragma once

#include <cfloat>
#include <sstream> 

#define EIGEN_MPL2_ONLY // this is for being sure of picking just the MPL2 licensed parts

#include <Eigen/Dense>
#include <Eigen/CholmodSupport>
#include <Eigen/SparseCore>

#include "SparseMatUtils.h"

#include "Log.h"
#include "Logger.h"

//#define CHOLMOD_SOLVER_DEBUG


///  \addtogroup cv_Sparse_Module
/* @{ */

///	\class CholmodSolver
///	\author Luigi Freda
///	\brief 
///	\note
/// 	\todo 
///	\date
///	\warning
class CholmodSolver 
{
public:
    
    CholmodSolver(int size):
    _msA(size,size),  // init _msA 
    _msAa(size,size),
    _msI(size,size),        
    _vdb(size) 
    {
        //_msI.setIdentity(); // only new version of Eigen 
        for(int j=0; j<_msI.cols(); j++)
        {
            _msI.coeffRef(j,j) = 1; 
        }
    }
    
    // solve on augmented matrix 
    bool Solve()
    {
        _solver.compute(_msAa);
        if (_solver.info() != Eigen::Success)
        {
            LogError << "CholmodSolver::Solve() - decomposition failed "; 
#ifdef CHOLMOD_SOLVER_DEBUG
            std::stringstream ss; 
            ss << "cholmod_failure_" << _iFileNum++; 
            LoggerFile file(ss.str() + ".txt");
            file << "sparse mat: " << _msAa.rows() << " x " << _msAa.cols() << std::endl; 
            file << _msAa;
            sparsematutils::savePatternEps(_msAa,ss.str() + ".eps");
#endif
            //_vdx.setZero(); 
            return false;
        }
        _vdx = _solver.solve(_vdb);
        if (_solver.info() != Eigen::Success)
        {
            LogError << "CholmodSolver::Solve() - solve failed "; 
            return false;
        }
        return true; 
    }
    
    
public: // setters 
    
    // reserve size for each inner vector of A by providing a vector object 
    // with an operator[](int j) returning the reserve size of the j-th inner vector
    void ReserveA(const std::vector<int>& sizes)
    {
        _msA.reserve(sizes); 
        _msAa.reserve(sizes); 
    }
    
    // add the buffer data to block (i,j,rows,cols) of A; the buffer data is assumed to represent a symmetric matrix!
    void AddSymBlock2A(int i, int j, int rows, int cols, double* dataSym)
    {
        sparsematutils::add2BlockColMajor(_msA,i,j,rows,cols,dataSym/* symmetric data accessed as col major order */);
    }
    
    void SetAugmentedA(double lambda)
    {
        std::vector<int> vecDegenerateElems; // we cannot use list of pointers since the method _msAa.coeffRef(j,j) possibly reallocate all the _msA!!!
        double dTrace = 0; 
        _msAa = _msA;
        double lambdap1 = 1 + lambda;
        for(int j=0; j<_msAa.cols(); j++)
        {
            double& elemjj = _msAa.coeffRef(j,j);   // this seems to be the fastest way: binary search in each column 
            dTrace += elemjj;
            if(elemjj < DBL_EPSILON){ vecDegenerateElems.push_back(j); }// NOTE: no fabs needed! elemjj shoudl be >=0 coming from a positive-definite matrix 
            elemjj *= lambdap1;
        }
        
        if(!vecDegenerateElems.empty())
        {
            LogWarning << "CholmodSolver::SetAugmentedA() - fixing degeneracy "; 
            double lmInitialValue = 1e-3 * dTrace/_msAa.cols(); // typical initial value for Levenberg-Marquardt (See Hartley-Zisserman pag 600) 1e-3 * dTrace /= _msAa.cols()
            for(int i=0; i<vecDegenerateElems.size(); i++)
            {
                int idx = vecDegenerateElems[i];
                _msAa.coeffRef(idx,idx)= lmInitialValue*(1 + lambda);    
            }
        }
    }
    
    void SetNegVb(double* data, int size)
    {
        assert(_vdb.size() == size); 
        for(int i=0; i<size; i++)  _vdb[i] = -data[i];
    }
    
    // removes all non zeros but keep allocated memory
    void ResetA()
    {
        _msA.setZero(); // removes all non zeros but keep allocated memory 
    }
    
    void MakeACompressed()
    {
        _msA.makeCompressed();
    }
    
public: // getters 
    
    Eigen::VectorXd& GetX() { return _vdx; }
    
    double* GetXData() { return _vdx.data(); }
    
    int GetSize() { return _msA.rows(); }
    
protected:

    Eigen::SparseMatrix<double> _msA;  // sparse matrix A
    Eigen::SparseMatrix<double> _msAa; // sparse matrix A augmented A + lamda*I
    Eigen::SparseMatrix<double> _msI;  // sparse matrix Identity
    Eigen::VectorXd  _vdb;  // dense vector b 
    Eigen::VectorXd  _vdx;  // dense vector x (result)
    
    Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>,Eigen::Upper > _solver; // Only the upper triangular part of the input matrix A is considered for solving. 
                                                                                    // The opposite triangle might either be empty or contain arbitrary values.
    
#ifdef CHOLMOD_SOLVER_DEBUG
    static int _iFileNum; // for saving files 
#endif
};

