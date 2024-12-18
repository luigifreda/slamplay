/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2008, Willow Garage, Inc.
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

#ifndef ROS_TIME_IMPL_H_INCLUDED
#define ROS_TIME_IMPL_H_INCLUDED

/*********************************************************************
** Headers
*********************************************************************/

#include <ros/platform.h>
#include <iostream>
#include <cmath>
#include <ros/exception.h>
#include <ros/time.h>
#include <boost/date_time/posix_time/posix_time.hpp>

/*********************************************************************
** Cross Platform Headers
*********************************************************************/

#if defined(_WIN32)
  #include <sys/timeb.h>
#else
  #include <sys/time.h>
#endif

namespace ros
{

  template<class T, class D>
  T& TimeBase<T, D>::fromNSec(uint64_t t)
  {
    uint64_t sec64 = 0;
    uint64_t nsec64 = t;

    normalizeSecNSec(sec64, nsec64);

    sec = static_cast<uint32_t>(sec64);
    nsec = static_cast<uint32_t>(nsec64);

    return *static_cast<T*>(this);
  }

  template<class T, class D>
  T& TimeBase<T, D>::fromSec(double t) {
      int64_t sec64 = static_cast<int64_t>(floor(t));
      if (sec64 < 0 || sec64 > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("Time is out of dual 32-bit range");
      sec = static_cast<uint32_t>(sec64);
      nsec = static_cast<uint32_t>(boost::math::round((t-sec) * 1e9));
      // avoid rounding errors
      sec += (nsec / 1000000000ul);
      nsec %= 1000000000ul;
      return *static_cast<T*>(this);
  }

  template<class T, class D>
  D TimeBase<T, D>::operator-(const T &rhs) const
  {
    D d;
    return d.fromNSec(toNSec() - rhs.toNSec());
  }

  template<class T, class D>
  T TimeBase<T, D>::operator-(const D &rhs) const
  {
    return *static_cast<const T*>(this) + ( -rhs);
  }

  template<class T, class D>
  T TimeBase<T, D>::operator+(const D &rhs) const
  {
    int64_t sec_sum  = static_cast<uint64_t>(sec) + static_cast<uint64_t>(rhs.sec);
    int64_t nsec_sum = static_cast<uint64_t>(nsec) + static_cast<uint64_t>(rhs.nsec);

    // Throws an exception if we go out of 32-bit range
    normalizeSecNSecUnsigned(sec_sum, nsec_sum);

    // now, it's safe to downcast back to uint32 bits
    return T(static_cast<uint32_t>(sec_sum), static_cast<uint32_t>(nsec_sum));
  }

  template<class T, class D>
  T& TimeBase<T, D>::operator+=(const D &rhs)
  {
    *this = *this + rhs;
    return *static_cast<T*>(this);
  }

  template<class T, class D>
  T& TimeBase<T, D>::operator-=(const D &rhs)
  {
    *this += (-rhs);
    return *static_cast<T*>(this);
  }

  template<class T, class D>
  bool TimeBase<T, D>::operator==(const T &rhs) const
  {
    return sec == rhs.sec && nsec == rhs.nsec;
  }

  template<class T, class D>
  bool TimeBase<T, D>::operator<(const T &rhs) const
  {
    if (sec < rhs.sec)
      return true;
    else if (sec == rhs.sec && nsec < rhs.nsec)
      return true;
    return false;
  }

  template<class T, class D>
  bool TimeBase<T, D>::operator>(const T &rhs) const
  {
    if (sec > rhs.sec)
      return true;
    else if (sec == rhs.sec && nsec > rhs.nsec)
      return true;
    return false;
  }

  template<class T, class D>
  bool TimeBase<T, D>::operator<=(const T &rhs) const
  {
    if (sec < rhs.sec)
      return true;
    else if (sec == rhs.sec && nsec <= rhs.nsec)
      return true;
    return false;
  }

  template<class T, class D>
  bool TimeBase<T, D>::operator>=(const T &rhs) const
  {
    if (sec > rhs.sec)
      return true;
    else if (sec == rhs.sec && nsec >= rhs.nsec)
      return true;
    return false;
  }

  template<class T, class D>
  boost::posix_time::ptime
  TimeBase<T, D>::toBoost() const
  {
    namespace pt = boost::posix_time;
#if defined(BOOST_DATE_TIME_HAS_NANOSECONDS)
    return pt::from_time_t(sec) + pt::nanoseconds(nsec);
#else
    return pt::from_time_t(sec) + pt::microseconds(nsec/1000);
#endif
  }
}

#endif // ROS_IMPL_TIME_H_INCLUDED
