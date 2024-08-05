
#pragma once

#include <iostream>
extern "C" {
#include <cholmod.h>
}
#include <assert.h>

namespace slamplay {

///	\class CholmodDenseMatrix
///	\author Luigi Freda
///	\brief Interface towards cholmod dense matrix
///	\note
/// 	\todo
///	\date
///	\warning
class CholmodDenseMatrix {
   public:
    // Create a vector initialized with zeros
    CholmodDenseMatrix(int rows, int cols, cholmod_common* c) {
        assert(rows > 0 && cols > 0);
        _cholM = cholmod_zeros(rows, cols, CHOLMOD_REAL, c);
        _cholC = c;
    }

    virtual ~CholmodDenseMatrix() {
        cholmod_free_dense(&_cholM, _cholC);
    }

    // copy constructor
    CholmodDenseMatrix(const CholmodDenseMatrix& other) {
        _cholC = other._cholC;
        _cholM = cholmod_zeros(other.GetRows(), other.GetCols(), CHOLMOD_REAL, _cholC);
        cholmod_copy_dense2(other._cholM /*src*/, _cholM /*dst*/, _cholC);
    }

    // assigment operator =
    CholmodDenseMatrix& operator=(const CholmodDenseMatrix& other) {
        if (this != &other) {  // Not necessary in this case but it is useful to don't forget it
            cholmod_copy_dense2(other._cholM /*src*/, _cholM /*dst*/, _cholC);
        }
        return *this;
    }

    // assigment operator =
    CholmodDenseMatrix& operator=(cholmod_dense* mat) {
        if (this->_cholM != mat) {  // Not necessary in this case but it is useful to don't forget it
            cholmod_copy_dense2(mat /*src*/, _cholM /*dst*/, _cholC);
        }
        return *this;
    }

   public:  // setters
    inline void Set(int row, int col, double val) { ((double*)_cholM->x)[row + col * (_cholM->d)] = val; }

   public:  // getters
    cholmod_dense* GetMat() { return _cholM; }

    /// Return the matrix number of rows
    int GetRows() const { return _cholM->nrow; }
    /// Return the matrix number of columns
    int GetCols() const { return _cholM->ncol; }

    inline double& Get(int row, int col) { return ((double*)_cholM->x)[row + col * (_cholM->d)]; }
    inline const double& Get(int row, int col) const { return ((double*)_cholM->x)[row + col * (_cholM->d)]; }

   protected:
    cholmod_dense* _cholM;   // choldmod dense matrix representation
    cholmod_common* _cholC;  // common workspace
};

std::ostream& operator<<(std::ostream& out, const CholmodDenseMatrix& matrix) {
    for (int i = 0; i < matrix.GetRows(); ++i)
    {
        out << "Row " << i << ":";
        for (int j = 0; j < matrix.GetCols(); ++j)
            out << " " << matrix.Get(i, j);
        out << "\n";
    }
    return out;
}

}  // namespace slamplay