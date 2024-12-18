/*
 * Copyright (c) 2008, Willow Garage, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Willow Garage, Inc. nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <vector>

#include <gtest/gtest.h>
#include <ros/rate.h>
#include <ros/time.h>

#if !defined(_WIN32)
#include <sys/time.h>
#endif

#include <boost/date_time/posix_time/ptime.hpp>

using namespace ros;

/// \todo All the tests in here that use randomized values are not unit tests, replace them

double epsilon = 1e-9;

void seed_rand()
{
  //Seed random number generator with current microseond count
#if !defined(_WIN32)
  timeval temp_time_struct;
  gettimeofday(&temp_time_struct,NULL);
  srand(temp_time_struct.tv_usec);
#else
  srand(time(nullptr));
#endif
};

void generate_rand_times(uint32_t range, uint64_t runs, std::vector<ros::Time>& values1, std::vector<ros::Time>& values2)
{
  seed_rand();
  values1.clear();
  values2.clear();
  values1.reserve(runs);
  values2.reserve(runs);
  for ( uint32_t i = 0; i < runs ; i++ )
  {
    values1.push_back(ros::Time( (rand() * range / RAND_MAX), (rand() * 1000000000ULL/RAND_MAX)));
    values2.push_back(ros::Time( (rand() * range / RAND_MAX), (rand() * 1000000000ULL/RAND_MAX)));
  }
}

void generate_rand_durations(uint32_t range, uint64_t runs, std::vector<ros::Duration>& values1, std::vector<ros::Duration>& values2)
{
  seed_rand();
  values1.clear();
  values2.clear();
  values1.reserve(runs * 4);
  values2.reserve(runs * 4);
  for ( uint32_t i = 0; i < runs ; i++ )
  {
    // positive durations
    values1.push_back(ros::Duration(  (rand() * range / RAND_MAX),  (rand() * 1000000000ULL/RAND_MAX)));
    values2.push_back(ros::Duration(  (rand() * range / RAND_MAX),  (rand() * 1000000000ULL/RAND_MAX)));
    EXPECT_GE(values1.back(), ros::Duration(0,0));
    EXPECT_GE(values2.back(), ros::Duration(0,0));

    // negative durations
    values1.push_back(ros::Duration( -(rand() * range / RAND_MAX), -(rand() * 1000000000ULL/RAND_MAX)));
    values2.push_back(ros::Duration( -(rand() * range / RAND_MAX), -(rand() * 1000000000ULL/RAND_MAX)));
    EXPECT_LE(values1.back(), ros::Duration(0,0));
    EXPECT_LE(values2.back(), ros::Duration(0,0));

    // positive and negative durations
    values1.push_back(ros::Duration(  (rand() * range / RAND_MAX),  (rand() * 1000000000ULL/RAND_MAX)));
    values2.push_back(ros::Duration( -(rand() * range / RAND_MAX), -(rand() * 1000000000ULL/RAND_MAX)));
    EXPECT_GE(values1.back(), ros::Duration(0,0));
    EXPECT_LE(values2.back(), ros::Duration(0,0));

    // negative and positive durations
    values1.push_back(ros::Duration( -(rand() * range / RAND_MAX), -(rand() * 1000000000ULL/RAND_MAX)));
    values2.push_back(ros::Duration(  (rand() * range / RAND_MAX),  (rand() * 1000000000ULL/RAND_MAX)));
    EXPECT_LE(values1.back(), ros::Duration(0,0));
    EXPECT_GE(values2.back(), ros::Duration(0,0));
  }
}

TEST(Time, size)
{
  ASSERT_EQ(sizeof(Time), 8);
  ASSERT_EQ(sizeof(Duration), 8);
}

TEST(Time, Comparitors)
{
  std::vector<ros::Time> v1;
  std::vector<ros::Time> v2;
  generate_rand_times(100, 1000, v1,v2);

  for (uint32_t i = 0; i < v1.size(); i++)
  {
    if (v1[i].sec * 1000000000ULL + v1[i].nsec < v2[i].sec * 1000000000ULL + v2[i].nsec)
    {
      EXPECT_LT(v1[i], v2[i]);
      //      printf("%f %d ", v1[i].toSec(), v1[i].sec * 1000000000ULL + v1[i].nsec);
      //printf("vs %f %d\n", v2[i].toSec(), v2[i].sec * 1000000000ULL + v2[i].nsec);
      EXPECT_LE(v1[i], v2[i]);
      EXPECT_NE(v1[i], v2[i]);
    }
    else if (v1[i].sec * 1000000000ULL + v1[i].nsec > v2[i].sec * 1000000000ULL + v2[i].nsec)
    {
      EXPECT_GT(v1[i], v2[i]);
      EXPECT_GE(v1[i], v2[i]);
      EXPECT_NE(v1[i], v2[i]);
    }
    else
    {
      EXPECT_EQ(v1[i], v2[i]);
      EXPECT_LE(v1[i], v2[i]);
      EXPECT_GE(v1[i], v2[i]);
    }

  }

}

TEST(Time, ToFromDouble)
{
  std::vector<ros::Time> v1;
  std::vector<ros::Time> v2;
  generate_rand_times(100, 1000, v1,v2);

  for (uint32_t i = 0; i < v1.size(); i++)
  {
    EXPECT_EQ(v1[i].toSec(), v1[i].fromSec(v1[i].toSec()).toSec());

  }

}

TEST(Time, RoundingError)
{
  double someInt = 1031.0; // some integer
  double t = std::nextafter(someInt, 0); // someint - epsilon
  // t should be 1031.000000

  ros::Time exampleTime;
  exampleTime.fromSec(t);

  // if rounded incorrectly, sec may be 1030
  // and nsec may be 1e9.
  EXPECT_EQ(exampleTime.sec, 1031);
  EXPECT_EQ(exampleTime.nsec, 0);
}

TEST(Time, OperatorPlus)
{
  Time t(100, 0);
  Duration d(100, 0);
  Time r = t + d;
  EXPECT_EQ(r.sec, 200UL);
  EXPECT_EQ(r.nsec, 0UL);

  t = Time(0, 100000UL);
  d = Duration(0, 100UL);
  r = t + d;
  EXPECT_EQ(r.sec, 0UL);
  EXPECT_EQ(r.nsec, 100100UL);

  t = Time(0, 0);
  d = Duration(10, 2000003000UL);
  r = t + d;
  EXPECT_EQ(r.sec, 12UL);
  EXPECT_EQ(r.nsec, 3000UL);
}

TEST(Time, OperatorMinus)
{
  Time t(100, 0);
  Duration d(100, 0);
  Time r = t - d;
  EXPECT_EQ(r.sec, 0UL);
  EXPECT_EQ(r.nsec, 0UL);

  t = Time(0, 100000UL);
  d = Duration(0, 100UL);
  r = t - d;
  EXPECT_EQ(r.sec, 0UL);
  EXPECT_EQ(r.nsec, 99900UL);

  t = Time(30, 0);
  d = Duration(10, 2000003000UL);
  r = t - d;
  EXPECT_EQ(r.sec, 17UL);
  EXPECT_EQ(r.nsec, 999997000ULL);
}

TEST(Time, OperatorPlusEquals)
{
  Time t(100, 0);
  Duration d(100, 0);
  t += d;
  EXPECT_EQ(t.sec, 200UL);
  EXPECT_EQ(t.nsec, 0UL);

  t = Time(0, 100000UL);
  d = Duration(0, 100UL);
  t += d;
  EXPECT_EQ(t.sec, 0UL);
  EXPECT_EQ(t.nsec, 100100UL);

  t = Time(0, 0);
  d = Duration(10, 2000003000UL);
  t += d;
  EXPECT_EQ(t.sec, 12UL);
  EXPECT_EQ(t.nsec, 3000UL);
}

TEST(Time, OperatorMinusEquals)
{
  Time t(100, 0);
  Duration d(100, 0);
  t -= d;
  EXPECT_EQ(t.sec, 0UL);
  EXPECT_EQ(t.nsec, 0UL);

  t = Time(0, 100000UL);
  d = Duration(0, 100UL);
  t -= d;
  EXPECT_EQ(t.sec, 0UL);
  EXPECT_EQ(t.nsec, 99900UL);

  t = Time(30, 0);
  d = Duration(10, 2000003000UL);
  t -= d;
  EXPECT_EQ(t.sec, 17UL);
  EXPECT_EQ(t.nsec, 999997000ULL);
}

TEST(Time, SecNSecConstructor)
{
  Time t(100, 2000003000UL);
  EXPECT_EQ(t.sec, 102UL);
  EXPECT_EQ(t.nsec, 3000UL);
}

TEST(Time, DontMungeStreamState)
{
  std::ostringstream oss;
  Time t(100, 2000003000UL);
  oss << std::setfill('N');
  oss << std::setw(13);
  oss << t;

  EXPECT_EQ(oss.width(), 13);
  EXPECT_EQ(oss.fill(), 'N');
}

TEST(Time, ToFromBoost)
{
  std::vector<ros::Time> v1;
  std::vector<ros::Time> v2;
  generate_rand_times(100, 1000, v1,v2);

  for (uint32_t i = 0; i < v1.size(); i++)
  {
    Time t = v1[i];
    // dont assume that nanosecond are available
    t.nsec = uint32_t(t.nsec / 1000.0) * 1000;
    boost::posix_time::ptime b = t.toBoost();
    Time tt = Time::fromBoost(b);
    EXPECT_EQ(t, tt);
  }
}

TEST(Time, CastFromDoubleExceptions)
{
  ros::Time::init();

  Time t1, t2, t3;
  // Valid values to cast, must not throw exceptions
  EXPECT_NO_THROW(t1.fromSec(4294967295.0));
  EXPECT_NO_THROW(t2.fromSec(4294967295.999));
  EXPECT_NO_THROW(t3.fromSec(0.0000001));

  // The next casts all incorrect.
  EXPECT_THROW(t1.fromSec(4294967296.0), std::runtime_error);
  EXPECT_THROW(t2.fromSec(-0.0001), std::runtime_error);
  EXPECT_THROW(t3.fromSec(-4294967296.0), std::runtime_error);
}

TEST(Time, OperatorMinusExceptions)
{
  ros::Time::init();

  Time t1(2147483648, 0);
  Time t2(2147483647, 999999999);
  Time t3(2147483647, 999999998);
  Time t4(4294967295, 999999999);
  Time t5(4294967295, 999999998);
  Time t6(0, 1);

  Duration d1(2147483647, 999999999);
  Duration d2(-2147483648, 0);
  Duration d3(-2147483648, 1);
  Duration d4(0, 1);

  EXPECT_NO_THROW(t1 - t2);
  EXPECT_NO_THROW(t3 - t2);
  EXPECT_NO_THROW(t4 - t5);

  EXPECT_NO_THROW(t1 - d1);
  EXPECT_NO_THROW(t5 - d1);

  EXPECT_THROW(t4 - t6, std::runtime_error);
  EXPECT_THROW(t4 - t3, std::runtime_error);

  EXPECT_THROW(t1 - d2, std::runtime_error);
  EXPECT_THROW(t2 - d2, std::runtime_error);
  EXPECT_THROW(t4 - d3, std::runtime_error);
}

TEST(Time, OperatorPlusExceptions)
{
  ros::Time::init();

  Time t1(2147483648, 0);
  Time t2(2147483647, 999999999);
  Time t4(4294967295, 999999999);
  Time t5(4294967295, 999999998);

  Duration d1(2147483647, 999999999);
  Duration d2(-2147483648, 1);
  Duration d3(0, 2);
  Duration d4(0, 1);

  EXPECT_NO_THROW(t2 + d2);
  EXPECT_NO_THROW(t1 + d1);

  EXPECT_THROW(t4 + d4, std::runtime_error);
  EXPECT_THROW(t4 + d1, std::runtime_error);
  EXPECT_THROW(t5 + d3, std::runtime_error);
}

/************************************* Duration Tests *****************/

TEST(Duration, Comparitors)
{
  std::vector<ros::Duration> v1;
  std::vector<ros::Duration> v2;
  generate_rand_durations(100, 1000, v1,v2);

  for (uint32_t i = 0; i < v1.size(); i++)
  {
    if (v1[i].sec * 1000000000LL + v1[i].nsec < v2[i].sec * 1000000000LL + v2[i].nsec)
    {
      EXPECT_LT(v1[i], v2[i]);
//      printf("%f %lld ", v1[i].toSec(), v1[i].sec * 1000000000LL + v1[i].nsec);
//      printf("vs %f %lld\n", v2[i].toSec(), v2[i].sec * 1000000000LL + v2[i].nsec);
      EXPECT_LE(v1[i], v2[i]);
      EXPECT_NE(v1[i], v2[i]);
    }
    else if (v1[i].sec * 1000000000LL + v1[i].nsec > v2[i].sec * 1000000000LL + v2[i].nsec)
    {
      EXPECT_GT(v1[i], v2[i]);
//      printf("%f %lld ", v1[i].toSec(), v1[i].sec * 1000000000LL + v1[i].nsec);
//      printf("vs %f %lld\n", v2[i].toSec(), v2[i].sec * 1000000000LL + v2[i].nsec);
      EXPECT_GE(v1[i], v2[i]);
      EXPECT_NE(v1[i], v2[i]);
    }
    else
    {
      EXPECT_EQ(v1[i], v2[i]);
      EXPECT_LE(v1[i], v2[i]);
      EXPECT_GE(v1[i], v2[i]);
    }

  }
}

TEST(Duration, ToFromSec)
{
  std::vector<ros::Duration> v1;
  std::vector<ros::Duration> v2;
  generate_rand_durations(100, 1000, v1,v2);

  for (uint32_t i = 0; i < v1.size(); i++)
  {
    EXPECT_EQ(v1[i].toSec(), v1[i].fromSec(v1[i].toSec()).toSec());
    EXPECT_GE(ros::Duration(v1[i].toSec()).nsec, 0);
  }

  EXPECT_EQ(ros::Duration(-0.5), ros::Duration(-1LL, 500000000LL));
  EXPECT_EQ(ros::Duration(-0.5), ros::Duration(0, -500000000LL));
}

TEST(Duration, FromNSec)
{
  ros::Duration t;
  t.fromNSec(-500000000LL);
  EXPECT_EQ(ros::Duration(-0.5), t);

  t.fromNSec(-1500000000LL);
  EXPECT_EQ(ros::Duration(-1.5), t);

  t.fromNSec(500000000LL);
  EXPECT_EQ(ros::Duration(0.5), t);

  t.fromNSec(1500000000LL);
  EXPECT_EQ(ros::Duration(1.5), t);
}

TEST(Duration, OperatorPlus)
{
  std::vector<ros::Duration> v1;
  std::vector<ros::Duration> v2;
  generate_rand_durations(100, 1000, v1,v2);

  for (uint32_t i = 0; i < v1.size(); i++)
  {
    EXPECT_NEAR(v1[i].toSec() + v2[i].toSec(), (v1[i] + v2[i]).toSec(), epsilon);
    ros::Duration temp = v1[i];
    EXPECT_NEAR(v1[i].toSec() + v2[i].toSec(), (temp += v2[i]).toSec(), epsilon);

  }

}

TEST(Duration, OperatorMinus)
{
  std::vector<ros::Duration> v1;
  std::vector<ros::Duration> v2;
  generate_rand_durations(100, 1000, v1,v2);

  for (uint32_t i = 0; i < v1.size(); i++)
  {
    EXPECT_NEAR(v1[i].toSec() - v2[i].toSec(), (v1[i] - v2[i]).toSec(), epsilon);
    ros::Duration temp = v1[i];
    EXPECT_NEAR(v1[i].toSec() - v2[i].toSec(), (temp -= v2[i]).toSec(), epsilon);

    EXPECT_NEAR(- v2[i].toSec(), (-v2[i]).toSec(), epsilon);

  }

  ros::Time t1(1.1);
  ros::Time t2(1.3);
  ros::Duration time_diff = t1 - t2; //=-0.2

  EXPECT_NEAR(time_diff.toSec(), -0.2, epsilon);
  EXPECT_LE(time_diff, ros::Duration(-0.19));
  EXPECT_GE(time_diff, ros::Duration(-0.21));
}

TEST(Duration, OperatorTimes)
{
  std::vector<ros::Duration> v1;
  std::vector<ros::Duration> v2;
  generate_rand_durations(100, 1000, v1,v2);

  for (uint32_t i = 0; i < v1.size(); i++)
  {
    EXPECT_NEAR(v1[i].toSec() * v2[i].toSec(), (v1[i] * v2[i].toSec()).toSec(), epsilon);
    ros::Duration temp = v1[i];
    EXPECT_NEAR(v1[i].toSec() * v2[i].toSec(), (temp *= v2[i].toSec()).toSec(), epsilon);

  }

}

TEST(Duration, OperatorPlusEquals)
{
  Duration t(100, 0);
  Duration d(100, 0);
  t += d;
  EXPECT_EQ(t.sec, 200L);
  EXPECT_EQ(t.nsec, 0L);

  t = Duration(0, 100000L);
  d = Duration(0, 100L);
  t += d;
  EXPECT_EQ(t.sec, 0L);
  EXPECT_EQ(t.nsec, 100100L);

  t = Duration(0, 0);
  d = Duration(10, 2000003000L);
  t += d;
  EXPECT_EQ(t.sec, 12L);
  EXPECT_EQ(t.nsec, 3000L);
}

TEST(Duration, OperatorMinusEquals)
{
  Duration t(100, 0);
  Duration d(100, 0);
  t -= d;
  EXPECT_EQ(t.sec, 0L);
  EXPECT_EQ(t.nsec, 0L);

  t = Duration(0, 100000L);
  d = Duration(0, 100L);
  t -= d;
  EXPECT_EQ(t.sec, 0L);
  EXPECT_EQ(t.nsec, 99900L);

  t = Duration(30, 0);
  d = Duration(10, 2000003000L);
  t -= d;
  EXPECT_EQ(t.sec, 17L);
  EXPECT_EQ(t.nsec, 999997000L);
}

void alarmHandler(int sig)
{

}

TEST(Duration, sleepWithSignal)
{
#if !defined(_WIN32)
  signal(SIGALRM, alarmHandler);
  alarm(1);
#endif

  Time start = Time::now();
  Duration d(2.0);
  bool rc = d.sleep();
  Time end = Time::now();

  ASSERT_GT(end - start, d);
  ASSERT_TRUE(rc);
}

TEST(Rate, constructFromDuration){
  Duration d(4, 0);
  Rate r(d);
  EXPECT_EQ(r.expectedCycleTime(), d);
}

TEST(Rate, sleep_return_value_true){
  Rate r(Duration(0.2));
  Duration(r.expectedCycleTime() * 0.5).sleep();
  EXPECT_TRUE(r.sleep());
}

TEST(Rate, sleep_return_value_false){
  Rate r(Duration(0.2));
  Duration(r.expectedCycleTime() * 2).sleep();
  EXPECT_FALSE(r.sleep());  // requested rate cannot be achieved
}

TEST(WallRate, constructFromDuration){
  Duration d(4, 0);
  WallRate r(d);
  WallDuration wd(4, 0);
  EXPECT_EQ(r.expectedCycleTime(), wd);
}

///////////////////////////////////////////////////////////////////////////////////
// WallTime/WallDuration
///////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////
// SteadyTime/WallDuration
///////////////////////////////////////////////////////////////////////////////////

TEST(SteadyTime, sleep){
  SteadyTime start = SteadyTime::now();
  WallDuration d(2.0);
  bool rc = d.sleep();
  SteadyTime end = SteadyTime::now();

  ASSERT_GT(end - start, d);
  ASSERT_TRUE(rc);
}

TEST(SteadyTime, sleepUntil){
  SteadyTime start = SteadyTime::now();
  SteadyTime end = start + WallDuration(2.0);
  bool rc = SteadyTime::sleepUntil(end);
  SteadyTime finished = SteadyTime::now();

  ASSERT_GT(finished, end);
  ASSERT_TRUE(rc);
}

int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  ros::Time::init();
  return RUN_ALL_TESTS();
}
