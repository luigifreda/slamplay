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
#include <list>

#include "g2o/types/slam3d/types_slam3d.h"
#include "g2o/types/slam3d_addons/plane3d.h"

using namespace g2o;
using namespace std;
using namespace Eigen;

ostream& operator<<(ostream& os, const Plane3D& p) {
  Eigen::Vector4d v = p.toVector();
  os << "coeffs: " << v[0] << " " << v[1] << " " << v[2] << " " << v[3];
  return os;
}

typedef std::list<Plane3D*> PlaneList;

void transform(PlaneList& l, const SE3Quat& t) {
  for (PlaneList::iterator it = l.begin(); it != l.end(); ++it) {
    Plane3D* p = *it;
    *p = t * (*p);
  }
}

ostream& operator<<(ostream& os, const PlaneList& l) {
  for (PlaneList::const_iterator it = l.begin(); it != l.end(); ++it) {
    const Plane3D* p = *it;
    os << *p << endl;
  }
  return os;
}

int main(int argc, char** argv) 
{
  Plane3D p;
  cerr << "p0  " << p << endl;
  Eigen::Vector4d v;
  v << 0.5, 0.2, 0.1, 10;
  Plane3D p1;
  p1.fromVector(v);
  cerr << "p1  " << p1 << endl;
  Plane3D p2 = p1;
  cerr << "p2  " << p2 << endl;

  cerr << "azimuth " << Plane3D::azimuth(p1.normal()) << endl;
  cerr << "elevation " << Plane3D::elevation(p1.normal()) << endl;
  Vector3d mv = p2.ominus(p1);
  cerr << "p ominus p " << mv[0] << " " << mv[1] << " " << mv[2] << endl;

  p1.fromVector(Eigen::Vector4d(2, 2, 100, 10));
  cerr << "p1  " << p1 << endl;
  cerr << "azimuth " << Plane3D::azimuth(p1.normal()) << endl;
  cerr << "elevation " << Plane3D::elevation(p1.normal()) << endl;
  p2.fromVector(Eigen::Vector4d(-2, -2, 100, 100));
  cerr << "p2  " << p2 << endl;
  cerr << "azimuth " << Plane3D::azimuth(p2.normal()) << endl;
  cerr << "elevation " << Plane3D::elevation(p2.normal()) << endl;

  mv = p2.ominus(p1);
  cerr << "p ominus p " << mv[0] << " " << mv[1] << " " << mv[2] << endl;

  Plane3D p3 = p1;
  cerr << "p3  " << p3 << endl;
  p3.oplus(mv);
  cerr << "p3.oplus(mv) " << p3 << endl;

  PlaneList l;
  Plane3D* pp = new Plane3D();
  Eigen::Vector4d coeffs;
  coeffs << 1., 0., 0., 1.;
  pp->fromVector(coeffs);
  l.push_back(pp);

  pp = new Plane3D();
  coeffs << 0., 1., 0., 1.;
  pp->fromVector(coeffs);
  l.push_back(pp);

  pp = new Plane3D();
  coeffs << 0., 0., 1., 1.;
  pp->fromVector(coeffs);
  l.push_back(pp);

  cerr << l << endl;

  AngleAxisd r(AngleAxisd(0.0, Vector3d::UnitZ()) *
               AngleAxisd(0., Vector3d::UnitY()) *
               AngleAxisd(0., Vector3d::UnitX()));

  SE3Quat t(r.toRotationMatrix(), Vector3d(0.9, 0, 0));
  cerr << t << endl;
  transform(l, t);
  cerr << l << endl;

  return 0;
}
