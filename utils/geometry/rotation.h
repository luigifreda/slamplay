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

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace slamplay {

template <typename T>
inline Eigen::Matrix3d enforceRotationMatrix(const Eigen::Matrix<T, 3, 3> &mat) {
    // Perform Singular Value Decomposition (SVD)
    Eigen::JacobiSVD<Eigen::Matrix<T, 3, 3>> svd(mat, Eigen::ComputeFullU | Eigen::ComputeFullV);

    // Correct the rotation matrix
    auto R = svd.matrixU() * svd.matrixV().transpose();

    // Ensure determinant is +1 (proper rotation)
    if (R.determinant() < 0) {
        const auto U = svd.matrixU();
        U.col(2) *= -1;  // Flip the sign of the last column
        R = U * svd.matrixV().transpose();
    }
    return R;
}

//////////////////////////////////////////////////////////////////
// math functions needed for rotation conversion.

// dot and cross production

template <typename T>
inline T dotProduct(const T x[3], const T y[3]) {
    return (x[0] * y[0] + x[1] * y[1] + x[2] * y[2]);
}

template <typename T>
inline void crossProduct(const T x[3], const T y[3], T result[3]) {
    result[0] = x[1] * y[2] - x[2] * y[1];
    result[1] = x[2] * y[0] - x[0] * y[2];
    result[2] = x[0] * y[1] - x[1] * y[0];
}

//////////////////////////////////////////////////////////////////

// Converts from a angle anxis to quaternion :
template <typename T>
inline void angleAxisToQuaternion(const T *angle_axis, T *quaternion) {
    const T &a0 = angle_axis[0];
    const T &a1 = angle_axis[1];
    const T &a2 = angle_axis[2];
    const T theta_squared = a0 * a0 + a1 * a1 + a2 * a2;

    if (theta_squared > T(std::numeric_limits<double>::epsilon())) {
        const T theta = sqrt(theta_squared);
        const T half_theta = theta * T(0.5);
        const T k = sin(half_theta) / theta;
        quaternion[0] = cos(half_theta);
        quaternion[1] = a0 * k;
        quaternion[2] = a1 * k;
        quaternion[3] = a2 * k;
    } else {  // in case if theta_squared is zero
        const T k(0.5);
        quaternion[0] = T(1.0);
        quaternion[1] = a0 * k;
        quaternion[2] = a1 * k;
        quaternion[3] = a2 * k;
    }
}

template <typename T>
inline void quaternionToAngleAxis(const T *quaternion, T *angle_axis) {
    const T &q1 = quaternion[1];
    const T &q2 = quaternion[2];
    const T &q3 = quaternion[3];
    const T sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3;

    // For quaternions representing non-zero rotation, the conversion
    // is numercially stable
    if (sin_squared_theta > T(std::numeric_limits<double>::epsilon())) {
        const T sin_theta = sqrt(sin_squared_theta);
        const T &cos_theta = quaternion[0];

        // If cos_theta is negative, theta is greater than pi/2, which
        // means that angle for the angle_axis vector which is 2 * theta
        // would be greater than pi...

        const T two_theta = T(2.0) * ((cos_theta < 0.0)
                                          ? atan2(-sin_theta, -cos_theta)
                                          : atan2(sin_theta, cos_theta));
        const T k = two_theta / sin_theta;

        angle_axis[0] = q1 * k;
        angle_axis[1] = q2 * k;
        angle_axis[2] = q3 * k;
    } else {
        // For zero rotation, sqrt() will produce NaN in derivative since
        // the argument is zero. By approximating with a Taylor series,
        // and truncating at one term, the value and first derivatives will be
        // computed correctly when Jets are used..
        const T k(2.0);
        angle_axis[0] = q1 * k;
        angle_axis[1] = q2 * k;
        angle_axis[2] = q3 * k;
    }
}

template <typename T>
inline void angleAxisRotatePoint(const T angle_axis[3], const T pt[3], T result[3]) {
    const T theta2 = dotProduct(angle_axis, angle_axis);
    if (theta2 > T(std::numeric_limits<double>::epsilon())) {
        // Away from zero, use the rodriguez formula
        //
        //   result = pt costheta +
        //            (w x pt) * sintheta +
        //            w (w . pt) (1 - costheta)
        //
        // We want to be careful to only evaluate the square root if the
        // norm of the angle_axis vector is greater than zero. Otherwise
        // we get a division by zero.
        //
        const T theta = sqrt(theta2);
        const T costheta = cos(theta);
        const T sintheta = sin(theta);
        const T theta_inverse = 1.0 / theta;

        const T w[3] = {angle_axis[0] * theta_inverse,
                        angle_axis[1] * theta_inverse,
                        angle_axis[2] * theta_inverse};

        // Explicitly inlined evaluation of the cross product for
        // performance reasons.
        /*const T w_cross_pt[3] = { w[1] * pt[2] - w[2] * pt[1],
                                  w[2] * pt[0] - w[0] * pt[2],
                                  w[0] * pt[1] - w[1] * pt[0] };*/
        T w_cross_pt[3];
        crossProduct(w, pt, w_cross_pt);

        const T tmp = dotProduct(w, pt) * (T(1.0) - costheta);
        //    (w[0] * pt[0] + w[1] * pt[1] + w[2] * pt[2]) * (T(1.0) - costheta);

        result[0] = pt[0] * costheta + w_cross_pt[0] * sintheta + w[0] * tmp;
        result[1] = pt[1] * costheta + w_cross_pt[1] * sintheta + w[1] * tmp;
        result[2] = pt[2] * costheta + w_cross_pt[2] * sintheta + w[2] * tmp;
    } else {
        // Near zero, the first order Taylor approximation of the rotation
        // matrix R corresponding to a vector w and angle w is
        //
        //   R = I + hat(w) * sin(theta)
        //
        // But sintheta ~ theta and theta * w = angle_axis, which gives us
        //
        //  R = I + hat(w)
        //
        // and actually performing multiplication with the point pt, gives us
        // R * pt = pt + w x pt.
        //
        // Switching to the Taylor expansion near zero provides meaningful
        // derivatives when evaluated using Jets.
        //
        // Explicitly inlined evaluation of the cross product for
        // performance reasons.
        /*const T w_cross_pt[3] = { angle_axis[1] * pt[2] - angle_axis[2] * pt[1],
                                  angle_axis[2] * pt[0] - angle_axis[0] * pt[2],
                                  angle_axis[0] * pt[1] - angle_axis[1] * pt[0] };*/
        T w_cross_pt[3];
        crossProduct(angle_axis, pt, w_cross_pt);

        result[0] = pt[0] + w_cross_pt[0];
        result[1] = pt[1] + w_cross_pt[1];
        result[2] = pt[2] + w_cross_pt[2];
    }
}

}  // namespace slamplay