^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package rostime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.7.2 (2020-05-28)
------------------
* [Windows] Using C++11 std::chrono for ros_walltime & ros_steadytime (`#121 <https://github.com/ros/roscpp_core/issues/121>`_)

0.7.1 (2020-01-25)
------------------
* only depend on the boost components needed (`#117 <https://github.com/ros/roscpp_core/issues/117>`_)

0.7.0 (2020-01-24)
------------------
* rostime: remove empty destructor of DurationBase (`#104 <https://github.com/ros/roscpp_core/issues/104>`_)
* various code cleanup (`#116 <https://github.com/ros/roscpp_core/issues/116>`_)
* Bump CMake version to avoid CMP0048 warning (`#115 <https://github.com/ros/roscpp_core/issues/115>`_)

0.6.13 (2019-10-03)
-------------------
* use _WIN32 for platform detection (`#110 <https://github.com/ros/roscpp_core/issues/110>`_)
* Clarified documentation for time validity (`#109 <https://github.com/ros/roscpp_core/issues/109>`_)
* rostime: replace c-style casts with static_casts (`#106 <https://github.com/ros/roscpp_core/issues/106>`_)

0.6.12 (2019-03-04)
-------------------
* use std::numeric_limits instead of * _MAX macros for range checking (`#103 <https://github.com/ros/roscpp_core/issues/103>`_)
* use std::this_thread::sleep_for instead of WaitableTimer (`#101 <https://github.com/ros/roscpp_core/issues/101>`_)
* include windows.h in time.cpp (`#100 <https://github.com/ros/roscpp_core/issues/100>`_)
* fix duration bug and add tests. (`#98 <https://github.com/ros/roscpp_core/issues/98>`_)
* fix for Duration::fromSec() which had rounding issues (`#93 <https://github.com/ros/roscpp_core/issues/93>`_)
* fix bug in HAVE_CXXABI_H compiler check (`#89 <https://github.com/ros/roscpp_core/issues/89>`_)
* add ROSTIME_DECL storage-class attribute (`#90 <https://github.com/ros/roscpp_core/issues/90>`_)

0.6.11 (2018-06-06)
-------------------
* argument to boost microseconds must be integral for Boost 1.67 and newer compatibility (`#79 <https://github.com/ros/roscpp_core/issues/79>`_)
* remove empty destructor of TimeBase (which makes TimeBase, Time and WallTime trivially copyable) (`#82 <https://github.com/ros/roscpp_core/issues/82>`_)

0.6.10 (2018-05-01)
-------------------
* fix conversion of Duration on macOS (`#78 <https://github.com/ros/roscpp_core/issues/78>`_)

0.6.9 (2018-02-02)
------------------
* expose ros_walltime and ros_steadytime (`#73 <https://github.com/ros/roscpp_core/issues/73>`_)

0.6.8 (2018-01-26)
------------------

0.6.7 (2017-11-03)
------------------

0.6.6 (2017-10-25)
------------------
* only use Apple features on Apple (`#68 <https://github.com/ros/roscpp_core/issues/68>`_)
* remove exception specification (`#64 <https://github.com/ros/roscpp_core/issues/64>`_)

0.6.5 (2017-07-27)
------------------
* add additional checks for valid time values (`#62 <https://github.com/ros/roscpp_core/pull/62>`_)
* fix overflow bugs in Time and Duration (`#61 <https://github.com/ros/roscpp_core/pull/61>`_, `#63 <https://github.com/ros/roscpp_core/pull/63>`_)

0.6.4 (2017-06-06)
------------------
* add logic to support steady time on macOS (regression of 0.6.3) (`#59 <https://github.com/ros/roscpp_core/pull/59>`_)

0.6.3 (2017-05-15)
------------------
* add SteadyTime (`#57 <https://github.com/ros/roscpp_core/issues/57>`_)

0.6.2 (2017-02-14)
------------------

0.6.1 (2016-09-02)
------------------
* fix rounding errors leading to invalid stored data in ros::TimeBase (`#48 <https://github.com/ros/roscpp_core/issues/48>`_)

0.6.0 (2016-03-17)
------------------
* change Duration:sleep return semantic (`#47 <https://github.com/ros/roscpp_core/pull/47>`_)

0.5.7 (2016-03-09)
------------------
* Adjust return value of sleep() function (`#45 <https://github.com/ros/roscpp_core/pull/45>`_)
* fix WallRate(Duration) constructor (`#40 <https://github.com/ros/roscpp_core/pull/40>`_)

0.5.6 (2015-05-20)
------------------

0.5.5 (2014-12-22)
------------------
* move implementation of Duration(Rate) constructor (`#30 <https://github.com/ros/roscpp_core/issues/30>`_)
* fix Duration initialization from seconds for negative values  (`#29 <https://github.com/ros/roscpp_core/pull/29>`_)

0.5.4 (2014-07-23)
------------------
* fix Rate initialized by Duration (`#26 <https://github.com/ros/roscpp_core/issues/26>`_)

0.5.3 (2014-06-28)
------------------

0.5.2 (2014-06-27)
------------------

0.5.1 (2014-06-24)
------------------

0.5.0 (2014-02-19)
------------------

0.4.2 (2014-02-11)
------------------

0.4.1 (2014-02-11)
------------------

0.4.0 (2014-01-29)
------------------

0.3.17 (2014-01-07)
-------------------
* fix boost include dir

0.3.16 (2013-07-14)
-------------------
* support for CATKIN_ENABLE_TESTING

0.3.15 (2013-06-06)
-------------------
* fix install destination for dll's under Windows

0.3.14 (2013-03-21)
-------------------

0.3.13 (2013-03-08)
-------------------

0.3.12 (2013-01-13)
-------------------
* improve string output of negative durations (`#3309 <https://github.com/ros/roscpp_core/issues/3309>`_)

0.3.11 (2012-12-21)
-------------------
* first public release for Groovy
