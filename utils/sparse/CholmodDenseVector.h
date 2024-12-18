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