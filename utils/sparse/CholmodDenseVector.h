#pragma once

#include "CholmodDenseMatrix.h"

namespace slamplay {

///	\class CholmodDenseVector
///	\author Luigi Freda
///	\brief Interface towards cholmod dense matrix (as a single column matrix)
///	\note
/// 	\todo
///	\date
///	\warning
class CholmodDenseVector : public CholmodDenseMatrix {
   public:
    // Create a vector initialized with zeros
    CholmodDenseVector(int size, cholmod_common* c) : CholmodDenseMatrix(size, 1, c)  // a vector as a single column matrix
    {}

   public:  // setters
    inline void Set(int i, double val) { ((double*)_cholM->x)[i] = val; }

   public:  // getters
    int GetSize() const { return GetRows(); }
    inline double& Get(int i) { return ((double*)_cholM->x)[i]; }
    inline const double& Get(int i) const { return ((double*)_cholM->x)[i]; }
};

}  // namespace slamplay