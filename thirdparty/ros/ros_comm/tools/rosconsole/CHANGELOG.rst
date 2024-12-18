^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package rosconsole
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.12.14 (2018-08-23)
--------------------

1.12.13 (2018-02-21)
--------------------

1.12.12 (2017-11-16)
--------------------

1.12.11 (2017-11-07)
--------------------

1.12.10 (2017-11-06)
--------------------

1.12.9 (2017-11-06)
-------------------

1.12.8 (2017-11-06)
-------------------
* replace 'while(0)' with 'while(false)' to avoid warnings (`#1179 <https://github.com/ros/ros_comm/issues/1179>`_)

1.12.7 (2017-02-17)
-------------------

1.12.6 (2016-10-26)
-------------------
* add missing walltime to roscpp logging (`#879 <https://github.com/ros/ros_comm/pull/879>`_)
* fix building on GCC-6 (`#911 <https://github.com/ros/ros_comm/pull/911>`_)

1.12.5 (2016-09-30)
-------------------

1.12.4 (2016-09-19)
-------------------

1.12.3 (2016-09-17)
-------------------

1.12.2 (2016-06-03)
-------------------

1.12.1 (2016-04-18)
-------------------
* use directory specific compiler flags (`#785 <https://github.com/ros/ros_comm/pull/785>`_)

1.12.0 (2016-03-18)
-------------------
* make LogAppender and Token destructor virtual (`#729 <https://github.com/ros/ros_comm/issues/729>`_)

1.11.18 (2016-03-17)
--------------------
* fix compiler warnings

1.11.17 (2016-03-11)
--------------------
* use boost::make_shared instead of new for constructing boost::shared_ptr (`#740 <https://github.com/ros/ros_comm/issues/740>`_)

1.11.16 (2015-11-09)
--------------------

1.11.15 (2015-10-13)
--------------------

1.11.14 (2015-09-19)
--------------------
* avoid redefining ROS_ASSERT_ENABLED (`#628 <https://github.com/ros/ros_comm/pull/628>`_)

1.11.13 (2015-04-28)
--------------------

1.11.12 (2015-04-27)
--------------------

1.11.11 (2015-04-16)
--------------------
* add DELAYED_THROTTLE versions of log macros (`#571 <https://github.com/ros/ros_comm/issues/571>`_)

1.11.10 (2014-12-22)
--------------------
* fix various defects reported by coverity

1.11.9 (2014-08-18)
-------------------

1.11.8 (2014-08-04)
-------------------

1.11.7 (2014-07-18)
-------------------

1.11.6 (2014-07-10)
-------------------

1.11.5 (2014-06-24)
-------------------
* rename variables within rosconsole macros (`#442 <https://github.com/ros/ros_comm/issues/442>`_)

1.11.4 (2014-06-16)
-------------------

1.11.3 (2014-05-21)
-------------------

1.11.2 (2014-05-08)
-------------------

1.11.1 (2014-05-07)
-------------------

1.11.0 (2014-03-04)
-------------------

1.10.0 (2014-02-11)
-------------------

1.9.54 (2014-01-27)
-------------------
* fix rosconsole segfault when using ROSCONSOLE_FORMAT with  (`#342 <https://github.com/ros/ros_comm/issues/342>`_)
* add missing run/test dependencies on rosbuild to get ROS_ROOT environment variable

1.9.53 (2014-01-14)
-------------------
* readd g_level_lockup symbol for backward compatibility when log4cxx is being used

1.9.52 (2014-01-08)
-------------------
* fix missing export of rosconsole backend interface library

1.9.51 (2014-01-07)
-------------------
* refactor rosconsole to not expose log4cxx, implement empty and log4cxx backends

1.9.50 (2013-10-04)
-------------------

1.9.49 (2013-09-16)
-------------------

1.9.48 (2013-08-21)
-------------------
* wrap condition in ROS_ASSERT_CMD in parenthesis (`#271 <https://github.com/ros/ros_comm/issues/271>`_)

1.9.47 (2013-07-03)
-------------------
* force CMake policy before setting preprocessor definition to ensure correct escaping (`#245 <https://github.com/ros/ros_comm/issues/245>`_)
* check for CATKIN_ENABLE_TESTING to enable configure without tests

1.9.46 (2013-06-18)
-------------------

1.9.45 (2013-06-06)
-------------------

1.9.44 (2013-03-21)
-------------------
* fix install destination for dll's under Windows

1.9.43 (2013-03-13)
-------------------

1.9.42 (2013-03-08)
-------------------
* fix handling spaces in folder names (`ros/catkin#375 <https://github.com/ros/catkin/issues/375>`_)

1.9.41 (2013-01-24)
-------------------

1.9.40 (2013-01-13)
-------------------
* fix dependent packages by pass LOG4CXX include dirs and libraries along
* fix usage of variable arguments in vFormatToBuffer() function

1.9.39 (2012-12-29)
-------------------
* first public release for Groovy
