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
#ifdef _MSC_VER
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
#endif

#include "ros/time.h"
#include "ros/impl/time.h"
#include <cmath>
#include <ctime>
#include <iomanip>
#include <limits>
#include <stdexcept>

// time related includes for macOS
#if defined(__APPLE__)
#include <mach/clock.h>
#include <mach/mach.h>
#endif  // defined(__APPLE__)

#ifdef _WINDOWS
#include <chrono>
#include <thread>
#include <windows.h>
#endif

#include <boost/thread/mutex.hpp>
#include <boost/io/ios_state.hpp>
#include <boost/date_time/posix_time/ptime.hpp>

/*********************************************************************
 ** Preprocessor
 *********************************************************************/

// Could probably do some better and more elaborate checking
// and definition here.
#define HAS_CLOCK_GETTIME (_POSIX_C_SOURCE >= 199309L)

/*********************************************************************
 ** Namespaces
 *********************************************************************/

namespace ros
{

  /*********************************************************************
   ** Variables
   *********************************************************************/

  const Duration DURATION_MAX(std::numeric_limits<int32_t>::max(), 999999999);
  const Duration DURATION_MIN(std::numeric_limits<int32_t>::min(), 0);

  const Time TIME_MAX(std::numeric_limits<uint32_t>::max(), 999999999);
  const Time TIME_MIN(0, 1);

  // This is declared here because it's set from the Time class but read from
  // the Duration class, and need not be exported to users of either.
  static bool g_stopped(false);

  // I assume that this is declared here, instead of time.h, to keep users
  // of time.h from including boost/thread/mutex.hpp
  static boost::mutex g_sim_time_mutex;

  static bool g_initialized(false);
  static bool g_use_sim_time(true);
  static Time g_sim_time(0, 0);

  /*********************************************************************
   ** Cross Platform Functions
   *********************************************************************/
  void ros_walltime(uint32_t& sec, uint32_t& nsec)
  {
#if !defined(_WIN32)
#if HAS_CLOCK_GETTIME
    timespec start;
    clock_gettime(CLOCK_REALTIME, &start);
    if (start.tv_sec < 0 || start.tv_sec > std::numeric_limits<uint32_t>::max())
      throw std::runtime_error("Timespec is out of dual 32-bit range");
    sec  = start.tv_sec;
    nsec = start.tv_nsec;
#else
    struct timeval timeofday;
    gettimeofday(&timeofday,NULL);
    if (timeofday.tv_sec < 0 || timeofday.tv_sec > std::numeric_limits<uint32_t>::max())
      throw std::runtime_error("Timeofday is out of dual signed 32-bit range");
    sec  = timeofday.tv_sec;
    nsec = timeofday.tv_usec * 1000;
#endif
#else
    uint64_t now_s = 0;
    uint64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    normalizeSecNSec(now_s, now_ns);

    sec = (uint32_t)now_s;
    nsec = (uint32_t)now_ns;
#endif
  }

  void ros_steadytime(uint32_t& sec, uint32_t& nsec)
  {
#if !defined(_WIN32)
    timespec start;
#if defined(__APPLE__)
    // On macOS use clock_get_time.
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    start.tv_sec = mts.tv_sec;
    start.tv_nsec = mts.tv_nsec;
#else  // defined(__APPLE__)
    // Otherwise use clock_gettime.
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif  // defined(__APPLE__)
    sec  = start.tv_sec;
    nsec = start.tv_nsec;
#else
    uint64_t now_s = 0;
    uint64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

    normalizeSecNSec(now_s, now_ns);

    sec = (uint32_t)now_s;
    nsec = (uint32_t)now_ns;
#endif
  }

  /*
   * These have only internal linkage to this translation unit.
   * (i.e. not exposed to users of the time classes)
   */

  /**
   * @brief Simple representation of the rt library nanosleep function.
   */
  int ros_nanosleep(const uint32_t &sec, const uint32_t &nsec)
  {
#if defined(_WIN32)
    std::this_thread::sleep_for(std::chrono::nanoseconds(static_cast<int64_t>(sec * 1e9 + nsec)));
    return 0;
#else
    timespec req = { sec, nsec };
    return nanosleep(&req, NULL);
#endif
  }

  /**
   * @brief Go to the wall!
   *
   * @todo Fully implement the win32 parts, currently just like a regular sleep.
   */
  bool ros_wallsleep(uint32_t sec, uint32_t nsec)
  {
#if defined(_WIN32)
    ros_nanosleep(sec,nsec);
#else
    timespec req = { sec, nsec };
    timespec rem = {0, 0};
    while (nanosleep(&req, &rem) && !g_stopped)
      {
        req = rem;
      }
#endif
    return !g_stopped;
  }

  /*********************************************************************
   ** Class Methods
   *********************************************************************/

  bool Time::useSystemTime()
  {
    return !g_use_sim_time;
  }

  bool Time::isSimTime()
  {
    return g_use_sim_time;
  }

  bool Time::isSystemTime()
  {
    return !isSimTime();
  }

  Time Time::now()
  {
    if (!g_initialized)
      {
        throw TimeNotInitializedException();
      }

    if (g_use_sim_time)
      {
        boost::mutex::scoped_lock lock(g_sim_time_mutex);
        Time t = g_sim_time;
        return t;
      }

    Time t;
    ros_walltime(t.sec, t.nsec);

    return t;
  }

  void Time::setNow(const Time& new_now)
  {
    boost::mutex::scoped_lock lock(g_sim_time_mutex);

    g_sim_time = new_now;
    g_use_sim_time = true;
  }

  void Time::init()
  {
    g_stopped = false;
    g_use_sim_time = false;
    g_initialized = true;
  }

  void Time::shutdown()
  {
    g_stopped = true;
  }

  bool Time::isValid()
  {
    return (!g_use_sim_time) || !g_sim_time.isZero();
  }

  bool Time::waitForValid()
  {
    return waitForValid(ros::WallDuration());
  }

  bool Time::waitForValid(const ros::WallDuration& timeout)
  {
    ros::WallTime start = ros::WallTime::now();
    while (!isValid() && !g_stopped)
    {
      ros::WallDuration(0.01).sleep();

      if (timeout > ros::WallDuration(0, 0) && (ros::WallTime::now() - start > timeout))
      {
        return false;
      }
    }

    if (g_stopped)
    {
      return false;
    }

    return true;
  }

  Time Time::fromBoost(const boost::posix_time::ptime& t)
  {
    boost::posix_time::time_duration diff = t - boost::posix_time::from_time_t(0);
    return Time::fromBoost(diff);
  }

  Time Time::fromBoost(const boost::posix_time::time_duration& d)
  {
    Time t;
    int64_t sec64 = d.total_seconds();
    if (sec64 < 0 || sec64 > std::numeric_limits<uint32_t>::max())
      throw std::runtime_error("time_duration is out of dual 32-bit range");
    t.sec = (uint32_t)sec64;
#if defined(BOOST_DATE_TIME_HAS_NANOSECONDS)
    t.nsec = d.fractional_seconds();
#else
    t.nsec = d.fractional_seconds()*1000;
#endif
    return t;
  }

  std::ostream& operator<<(std::ostream& os, const Time &rhs)
  {
    boost::io::ios_all_saver s(os);
    os << rhs.sec << "." << std::setw(9) << std::setfill('0') << rhs.nsec;
    return os;
  }

  std::ostream& operator<<(std::ostream& os, const Duration& rhs)
  {
    boost::io::ios_all_saver s(os);
    if (rhs.sec >= 0 || rhs.nsec == 0)
    {
      os << rhs.sec << "." << std::setw(9) << std::setfill('0') << rhs.nsec;
    }
    else
    {
      os << (rhs.sec == -1 ? "-" : "") << (rhs.sec + 1) << "." << std::setw(9) << std::setfill('0') << (1000000000 - rhs.nsec);
    }
    return os;
  }

  bool Time::sleepUntil(const Time& end)
  {
    if (Time::useSystemTime())
    {
      Duration d(end - Time::now());
      if (d > Duration(0))
      {
        return d.sleep();
      }

      return true;
    }
    else
    {
      Time start = Time::now();
      while (!g_stopped && (Time::now() < end))
      {
        ros_nanosleep(0,1000000);
        if (Time::now() < start)
        {
          return false;
        }
      }

      return true;
    }
  }

  bool WallTime::sleepUntil(const WallTime& end)
  {
    WallDuration d(end - WallTime::now());
    if (d > WallDuration(0))
      {
        return d.sleep();
      }

    return true;
  }

  bool SteadyTime::sleepUntil(const SteadyTime& end)
  {
    WallDuration d(end - SteadyTime::now());
    if (d > WallDuration(0))
    {
      return d.sleep();
    }

    return true;
  }

  bool Duration::sleep() const
  {
    if (Time::useSystemTime())
    {
      return ros_wallsleep(sec, nsec);
    }
    else
    {
      Time start = Time::now();
      Time end = start + *this;
      if (start.isZero())
      {
        end = TIME_MAX;
      }

      bool rc = false;
      while (!g_stopped && (Time::now() < end))
      {
        ros_wallsleep(0, 1000000);
        rc = true;

        // If we started at time 0 wait for the first actual time to arrive before starting the timer on
        // our sleep
        if (start.isZero())
        {
          start = Time::now();
          end = start + *this;
        }

        // If time jumped backwards from when we started sleeping, return immediately
        if (Time::now() < start)
        {
          return false;
        }
      }

      return rc && !g_stopped;
    }
  }

  std::ostream &operator<<(std::ostream& os, const WallTime &rhs)
  {
    boost::io::ios_all_saver s(os);
    os << rhs.sec << "." << std::setw(9) << std::setfill('0') << rhs.nsec;
    return os;
  }

  std::ostream &operator<<(std::ostream& os, const SteadyTime &rhs)
  {
    boost::io::ios_all_saver s(os);
    os << rhs.sec << "." << std::setw(9) << std::setfill('0') << rhs.nsec;
    return os;
  }

  WallTime WallTime::now()
  {
    WallTime t;
    ros_walltime(t.sec, t.nsec);

    return t;
  }

  SteadyTime SteadyTime::now()
  {
    SteadyTime t;
    ros_steadytime(t.sec, t.nsec);

    return t;
  }

  std::ostream &operator<<(std::ostream& os, const WallDuration& rhs)
  {
    boost::io::ios_all_saver s(os);
    if (rhs.sec >= 0 || rhs.nsec == 0)
    {
      os << rhs.sec << "." << std::setw(9) << std::setfill('0') << rhs.nsec;
    }
    else
    {
      os << (rhs.sec == -1 ? "-" : "") << (rhs.sec + 1) << "." << std::setw(9) << std::setfill('0') << (1000000000 - rhs.nsec);
    }
    return os;
  }

  bool WallDuration::sleep() const
  {
    return ros_wallsleep(sec, nsec);
  }

  void normalizeSecNSec(uint64_t& sec, uint64_t& nsec)
  {
    uint64_t nsec_part = nsec % 1000000000UL;
    uint64_t sec_part = nsec / 1000000000UL;

    if (sec + sec_part > std::numeric_limits<uint32_t>::max())
      throw std::runtime_error("Time is out of dual 32-bit range");

    sec += sec_part;
    nsec = nsec_part;
  }

  void normalizeSecNSec(uint32_t& sec, uint32_t& nsec)
  {
    uint64_t sec64 = sec;
    uint64_t nsec64 = nsec;

    normalizeSecNSec(sec64, nsec64);

    sec = (uint32_t)sec64;
    nsec = (uint32_t)nsec64;
  }

  void normalizeSecNSecUnsigned(int64_t& sec, int64_t& nsec)
  {
    int64_t nsec_part = nsec % 1000000000L;
    int64_t sec_part = sec + nsec / 1000000000L;
    if (nsec_part < 0)
      {
        nsec_part += 1000000000L;
        --sec_part;
      }

    if (sec_part < 0 || sec_part > std::numeric_limits<uint32_t>::max())
      throw std::runtime_error("Time is out of dual 32-bit range");

    sec = sec_part;
    nsec = nsec_part;
  }

  template class TimeBase<Time, Duration>;
  template class TimeBase<WallTime, WallDuration>;
  template class TimeBase<SteadyTime, WallDuration>;
}
