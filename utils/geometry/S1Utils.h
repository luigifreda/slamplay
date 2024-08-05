#pragma once

#include <math.h>
#include "SignalUtils.h"
#include "macros.h"

///\def M_2PI
#ifndef M_2PI
#define M_2PI 6.28318530717959  // 2*M_PI
#endif

namespace slamplay {

///	\class S1Utils
///	\brief some  manifold S1 (unit circle) utilities gathered in a class
///	\author
///	\note
class S1Utils {
   public:
    // positive modulus: similar to matlab's mod(), result is always positive. not similar to fmod()
    // es: posMod(-3,4)= 1   fmod(-3,4)= -3
    //     posMod(-5,4)= 3   fmod(-5,4)= -1
    static double posMod(double x, double y);

    // wraps an angle [rad] so that it is contained in [-M_PI,M_PI)
    static double wrap(double ang);

    // wraps an angle [deg] so that it is contained in [-180,180)
    static double wrapDeg(double ang);

    // wraps an angle [rad] so that it is contained in [0,2*M_PI)
    static double wrapPos(double ang);

    // wraps an angle [deg] so that it is contained in [0,360)
    static double wrapPosDeg(double ang);

    // returns the positive distance between ang1 [rad] and ang2 [rad] in the manifold S1 (unit circle)
    // result is smallest positive angle between ang1 and ang2
    static double distS1(double ang2, double ang1);

    // returns the positive distance between ang1 [deg] and ang2 [deg] in the manifold S1 (unit circle)
    // result is smallest positive angle between ang1 and ang2
    static double distS1Deg(double ang2, double ang1);

    // returns the difference ang2 [rad] - ang1 [rad] in the manifold S1 (unit circle)
    // result is the representation of the angle with the smallest absolute value
    static double diffS1(double ang2, double ang1);

    // returns the difference ang2 [deg] - ang1 [deg] in the manifold S1 (unit circle)
    // result is the representation of the angle with the smallest absolute value
    static double diffS1Deg(double ang2, double ang1);

    // check if the angle [deg] a is respectively between angle a [deg] and angle b [deg]
    static bool isAngleBetweenDeg(double angle, double a, double b);
    // check if the angle [deg] a is respectively between angle a [deg] and angle b [deg]
    // angles are assumed to be represented in [0,360°]
    static bool isAngleBetweenWithoutWrapDeg(double angle, double a, double b);

    // intersect the two angular intervals [a,b] and [c,d]: [resa,resb] is the result. Return true if there is intersection, false otherwise. Angles are represented in [deg]
    static bool intersectAngularIntervalsDeg(double& resa, double& resb, double a, double b, double c, double d);
};

// positive modulus: similar to matlab's mod(), result is always positive. not similar to fmod()
// es: posMod(-3,4)= 1   fmod(-3,4)= -3
//     posMod(-5,4)= 3   fmod(-5,4)= -1
inline double S1Utils::posMod(double x, double y) {
    if (y == 0) return x;

    return x - y * floor(x / y);
}

// wraps an angle [rad] so that it is contained in [-M_PI,M_PI)
inline double S1Utils::wrap(double ang) {
    /*ang = fmod( ang + M_PI, M_2PI ) - M_PI;  // now ang is in (-3*M_PI,M_PI)
    return ( (ang >= -M_PI) ? ang : (ang + M_2PI)  );*/
    return posMod(ang + M_PI, M_2PI) - M_PI;
}

// wraps an angle [deg] so that it is contained in [-180,180)
inline double S1Utils::wrapDeg(double ang) {
    /*ang = fmod( ang + 180., 360. ) - 180.;  // now ang is in (-3*180.,180.)
    return ( (ang >= -180.) ? ang : (ang + 360.)  );*/
    return posMod(ang + 180., 360.) - 180.;
}

// wraps an angle [rad] so that it is contained in [0,2*M_PI)
inline double S1Utils::wrapPos(double ang) {
    return posMod(ang, M_2PI);
}

// wraps an angle [deg] so that it is contained in [0,360)
inline double S1Utils::wrapPosDeg(double ang) {
    return posMod(ang, 360.);

    // ang = fmod( ang, 360. );  // now ang is in (-360.,360.)
    // return ( (ang <0.) ? (ang + 360.): ang  );
}

// returns the positive distance between ang1 [rad] and ang2 [rad] in the manifold S1 (unit circle)
// result is smallest positive angle between ang1 and ang2
inline double S1Utils::distS1(double ang2, double ang1) {
    double fabsdiff = fabs(fmod(ang2 - ang1, M_2PI));  // now fabsdiff is in (0,2*M_PI)
    return std::min(fabsdiff, M_2PI - fabsdiff);
}

// returns the positive distance between ang1 [deg] and ang2 [deg] in the manifold S1 (unit circle)
// result is smallest positive angle between ang1 and ang2
inline double S1Utils::distS1Deg(double ang2, double ang1) {
    double fabsdiff = fabs(fmod(ang2 - ang1, 360.));  // now fabsdiff is in (0.,360.)
    return std::min(fabsdiff, 360. - fabsdiff);
}

// returns the difference between ang1 [rad] and ang2 [rad] in the manifold S1 (unit circle)
// result is the representation of the angle with the smallest absolute value
inline double S1Utils::diffS1(double ang2, double ang1) {
    double diff = fmod(ang2 - ang1, M_2PI);  // now diff is in (-2*M_PI,2*M_PI)

    if (fabs(diff) <= M_PI)
        return diff;
    else
        return (diff - SignalUtils::sign(diff) * M_2PI);
}

// returns the difference between ang1 [deg] and ang2 [deg] in the manifold S1 (unit circle)
// result is the representation of the angle with the smallest absolute value
inline double S1Utils::diffS1Deg(double ang2, double ang1) {
    double diff = fmod(ang2 - ang1, 360);  // now diff is in (-360,360)

    if (fabs(diff) <= 180)
        return diff;
    else
        return (diff - SignalUtils::sign(diff) * 360);
}

// check if the angle [deg] a is respectively between angle a [deg] and angle b [deg]
inline bool S1Utils::isAngleBetweenDeg(double angle, double a, double b) {
    angle = wrapPosDeg(angle);  // angle is now in [0,360]
    a = wrapPosDeg(a);          // a is now in [0,360]
    b = wrapPosDeg(b);          // b is now in [0,360]

    return isAngleBetweenWithoutWrapDeg(angle, a, b);
}

// check if the angle [deg] a is respectively between angle a [deg] and angle b [deg]
// angles are assumed to be represented in [0,360°]
inline bool S1Utils::isAngleBetweenWithoutWrapDeg(double angle, double a, double b) {
    if (a < b)
    {
        return (a <= angle) && (angle <= b);
    } else
    {
        return (a <= angle) || (angle <= b);
    }
}

// intersect the two angular intervals [a,b] and [c,d]: [resa,resb] is the result. Return true if there is intersection, false otherwise. Angles are represented in [deg]
inline bool S1Utils::intersectAngularIntervalsDeg(double& resa, double& resb, double a, double b, double c, double d) {
    a = wrapPosDeg(a);  // a is now in [0,360]
    b = wrapPosDeg(b);  // b is now in [0,360]
    c = wrapPosDeg(c);  // c is now in [0,360]
    d = wrapPosDeg(d);  // d is now in [0,360]

    // cout << "[a,b] = ["<< a <<", "<<b<<"] " << "[c,d] = ["<<c<< ", "<<d<<"]" << endl;

    if (isAngleBetweenWithoutWrapDeg(c, a, b))
    {
        resa = c;
        // cout << "c is contained in [a,b]" << endl;
        if (isAngleBetweenWithoutWrapDeg(d, a, b))
        {
            // cout << "d is contained in [a,b]" << endl;
            resb = d;
        } else
        {
            resb = b;
        }
    } else
    {
        // cout << "c is not contained in [a,b]" << endl;
        if (isAngleBetweenWithoutWrapDeg(d, a, b))
        {
            // cout << "d is contained in [a,b]" << endl;
            resa = a;
            resb = d;
        } else  // both c and d do not belong to [a,b]
        {
            // cout << "d is not contained in [a,b]" << endl;
            if (isAngleBetweenWithoutWrapDeg(a, c, d))  // [a,b] is contained in [c,d]
            {
                // cout << "[a,b] is contained in [c,d]" << endl;
                resa = a;
                resb = b;
            } else
            {
                return false;
            }
        }
    }

    return true;
}

}  // namespace slamplay