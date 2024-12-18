/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2010, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/
#ifndef ROSTIME_IMPL_DURATION_H_INCLUDED
#define ROSTIME_IMPL_DURATION_H_INCLUDED

#include <ros/duration.h>
#include <ros/rate.h>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/math/special_functions/round.hpp>

namespace ros {
  //
  // DurationBase template member function implementation
  //
  template<class T>
  DurationBase<T>::DurationBase(int32_t _sec, int32_t _nsec)
  : sec(_sec), nsec(_nsec)
  {
    normalizeSecNSecSigned(sec, nsec);
  }

  template<class T>
  T& DurationBase<T>::fromSec(double d)
  {
    int64_t sec64 = static_cast<int64_t>(floor(d));
    if (sec64 < std::numeric_limits<int32_t>::min() || sec64 > std::numeric_limits<int32_t>::max())
      throw std::runtime_error("Duration is out of dual 32-bit range");
    sec = static_cast<int32_t>(sec64);
    nsec = static_cast<int32_t>(boost::math::round((d - sec) * 1e9));
    int32_t rollover = nsec / 1000000000ul;
    sec += rollover;
    nsec %= 1000000000ul;
    return *static_cast<T*>(this);
  }

  template<class T>
  T& DurationBase<T>::fromNSec(int64_t t)
  {
    int64_t sec64 = t / 1000000000LL;
    if (sec64 < std::numeric_limits<int32_t>::min() || sec64 > std::numeric_limits<int32_t>::max())
      throw std::runtime_error("Duration is out of dual 32-bit range");
    sec = static_cast<int32_t>(sec64);
    nsec = static_cast<int32_t>(t % 1000000000LL);

    normalizeSecNSecSigned(sec, nsec);

    return *static_cast<T*>(this);
  }

  template<class T>
  T DurationBase<T>::operator+(const T &rhs) const
  {
    T t;
    return t.fromNSec(toNSec() + rhs.toNSec());
  }

  template<class T>
  T DurationBase<T>::operator*(double scale) const
  {
    return T(toSec() * scale);
  }

  template<class T>
  T DurationBase<T>::operator-(const T &rhs) const
  {
    T t;
    return t.fromNSec(toNSec() - rhs.toNSec());
  }

  template<class T>
  T DurationBase<T>::operator-() const
  {
    T t;
    return t.fromNSec(-toNSec());
  }

  template<class T>
  T& DurationBase<T>::operator+=(const T &rhs)
  {
    *this = *this + rhs;
    return *static_cast<T*>(this);
  }

  template<class T>
  T& DurationBase<T>::operator-=(const T &rhs)
  {
    *this += (-rhs);
    return *static_cast<T*>(this);
  }

  template<class T>
  T& DurationBase<T>::operator*=(double scale)
  {
    fromSec(toSec() * scale);
    return *static_cast<T*>(this);
  }

  template<class T>
  bool DurationBase<T>::operator<(const T &rhs) const
  {
    if (sec < rhs.sec)
      return true;
    else if (sec == rhs.sec && nsec < rhs.nsec)
      return true;
    return false;
  }

  template<class T>
  bool DurationBase<T>::operator>(const T &rhs) const
  {
    if (sec > rhs.sec)
      return true;
    else if (sec == rhs.sec && nsec > rhs.nsec)
      return true;
    return false;
  }

  template<class T>
  bool DurationBase<T>::operator<=(const T &rhs) const
  {
    if (sec < rhs.sec)
      return true;
    else if (sec == rhs.sec && nsec <= rhs.nsec)
      return true;
    return false;
  }

  template<class T>
  bool DurationBase<T>::operator>=(const T &rhs) const
  {
    if (sec > rhs.sec)
      return true;
    else if (sec == rhs.sec && nsec >= rhs.nsec)
      return true;
    return false;
  }

  template<class T>
  bool DurationBase<T>::operator==(const T &rhs) const
  {
    return sec == rhs.sec && nsec == rhs.nsec;
  }

  template<class T>
  bool DurationBase<T>::isZero() const
  {
    return sec == 0 && nsec == 0;
  }

  template <class T>
  boost::posix_time::time_duration
  DurationBase<T>::toBoost() const
  {
    namespace bt = boost::posix_time;
#if defined(BOOST_DATE_TIME_HAS_NANOSECONDS)
    return bt::seconds(sec) + bt::nanoseconds(nsec);
#else
    return bt::seconds(sec) + bt::microseconds(nsec/1000);
#endif
  }
}
#endif
