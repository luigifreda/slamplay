^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package rosbag
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.12.14 (2018-08-23)
--------------------
* add TransportHint options --tcpnodelay and --udp (`#1295 <https://github.com/ros/ros_comm/issues/1295>`_)
* fix check for header first in rosbag play for rate control topic (`#1352 <https://github.com/ros/ros_comm/issues/1352>`_)

1.12.13 (2018-02-21)
--------------------
* return an error status on error in rosbag (`#1257 <https://github.com/ros/ros_comm/issues/1257>`_)
* fix warn of --max-splits without --split (`#1237 <https://github.com/ros/ros_comm/issues/1237>`_)

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
* fix Python 3 compatibility (`#1150 <https://github.com/ros/ros_comm/issues/1150>`_)
* fix handling connections without indices (`#1109 <https://github.com/ros/ros_comm/pull/1109>`_)
* improve message of check command (`#1067 <https://github.com/ros/ros_comm/pull/1067>`_)
* fix BZip2 inclusion (`#1016 <https://github.com/ros/ros_comm/pull/1016>`_)

1.12.7 (2017-02-17)
-------------------
* throw exception instead of accessing invalid memory (`#971 <https://github.com/ros/ros_comm/pull/971>`_)
* move headers to include/xmlrpcpp (`#962 <https://github.com/ros/ros_comm/issues/962>`_)
* added option wait-for-subscriber to rosbag play (`#959 <https://github.com/ros/ros_comm/issues/959>`_)
* terminate underlying rosbag play, record  on SIGTERM (`#951 <https://github.com/ros/ros_comm/issues/951>`_)
* add pause service for rosbag player (`#949 <https://github.com/ros/ros_comm/issues/949>`_)
* add rate-control-topic and rate-control-max-delay. (`#947 <https://github.com/ros/ros_comm/issues/947>`_)

1.12.6 (2016-10-26)
-------------------
* fix BagMigrationException in migrate_raw (`#917 <https://github.com/ros/ros_comm/issues/917>`_)

1.12.5 (2016-09-30)
-------------------

1.12.4 (2016-09-19)
-------------------

1.12.3 (2016-09-17)
-------------------
* set default values for min_space and min_space_str (`#883 <https://github.com/ros/ros_comm/issues/883>`_)
* record a maximum number of splits and then begin deleting old files (`#866 <https://github.com/ros/ros_comm/issues/866>`_)
* allow 64-bit sizes to be passed to robag max_size (`#865 <https://github.com/ros/ros_comm/issues/865>`_)
* update rosbag filter progress meter to use raw uncompressed input size (`#857 <https://github.com/ros/ros_comm/issues/857>`_)

1.12.2 (2016-06-03)
-------------------

1.12.1 (2016-04-18)
-------------------
* promote the result of read_messages to a namedtuple (`#777 <https://github.com/ros/ros_comm/pull/777>`_)
* use directory specific compiler flags (`#785 <https://github.com/ros/ros_comm/pull/785>`_)

1.12.0 (2016-03-18)
-------------------
* add missing parameter to AdvertiseOptions::createAdvertiseOptions (`#733 <https://github.com/ros/ros_comm/issues/733>`_)

1.11.18 (2016-03-17)
--------------------

1.11.17 (2016-03-11)
--------------------
* use boost::make_shared instead of new for constructing boost::shared_ptr (`#740 <https://github.com/ros/ros_comm/issues/740>`_)

1.11.16 (2015-11-09)
--------------------
* show size unit for --size of rosbag record in help string (`#697 <https://github.com/ros/ros_comm/pull/697>`_)

1.11.15 (2015-10-13)
--------------------
* add option --prefix for prefixing output topics (`#626 <https://github.com/ros/ros_comm/pull/626>`_)

1.11.14 (2015-09-19)
--------------------
* reduce memory usage by using slots for IndexEntry types (`#613 <https://github.com/ros/ros_comm/pull/613>`_)
* remove duplicate topics (`#647 <https://github.com/ros/ros_comm/issues/647>`_)
* better exception when calling get_start_time / get_end_time on empty bags (`#657 <https://github.com/ros/ros_comm/pull/657>`_)
* make support for lz4 in rosbag optional (`#642 <https://github.com/ros/ros_comm/pull/642>`_)
* fix handling of "play --topics" (`#620 <https://github.com/ros/ros_comm/issues/620>`_)

1.11.13 (2015-04-28)
--------------------

1.11.12 (2015-04-27)
--------------------

1.11.11 (2015-04-16)
--------------------
* add support for pausing when specified topics are about to be published (`#569 <https://github.com/ros/ros_comm/pull/569>`_)

1.11.10 (2014-12-22)
--------------------
* add option to specify the minimum disk space at which recording is stopped (`#500 <https://github.com/ros/ros_comm/pull/500>`_)
* add convenience API to Python rosbag (`#508 <https://github.com/ros/ros_comm/issues/508>`_)
* fix delay on detecting a running rosmaster with use_sim_time set (`#532 <https://github.com/ros/ros_comm/pull/532>`_)

1.11.9 (2014-08-18)
-------------------

1.11.8 (2014-08-04)
-------------------

1.11.7 (2014-07-18)
-------------------

1.11.6 (2014-07-10)
-------------------
* fix rosbag record prefix (`#449 <https://github.com/ros/ros_comm/issues/449>`_)

1.11.5 (2014-06-24)
-------------------
* Fix typo in rosbag usage

1.11.4 (2014-06-16)
-------------------
* Python 3 compatibility (`#426 <https://github.com/ros/ros_comm/issues/426>`_, `#430 <https://github.com/ros/ros_comm/issues/430>`_)

1.11.3 (2014-05-21)
-------------------

1.11.2 (2014-05-08)
-------------------

1.11.1 (2014-05-07)
-------------------
* add lz4 compression to rosbag (Python and C++) (`#356 <https://github.com/ros/ros_comm/issues/356>`_)
* fix rosbag record --node (`#357 <https://github.com/ros/ros_comm/issues/357>`_)
* move rosbag dox to rosbag_storage (`#389 <https://github.com/ros/ros_comm/issues/389>`_)

1.11.0 (2014-03-04)
-------------------
* use catkin_install_python() to install Python scripts (`#361 <https://github.com/ros/ros_comm/issues/361>`_)

1.10.0 (2014-02-11)
-------------------
* remove use of __connection header

1.9.54 (2014-01-27)
-------------------
* readd missing declaration of rosbag::createAdvertiseOptions (`#338 <https://github.com/ros/ros_comm/issues/338>`_)

1.9.53 (2014-01-14)
-------------------

1.9.52 (2014-01-08)
-------------------

1.9.51 (2014-01-07)
-------------------
* move several client library independent parts from ros_comm into roscpp_core, split rosbag storage specific stuff from client library usage (`#299 <https://github.com/ros/ros_comm/issues/299>`_)
* fix return value on platforms where char is unsigned.
* fix usage of boost include directories

1.9.50 (2013-10-04)
-------------------
* add chunksize option to rosbag record

1.9.49 (2013-09-16)
-------------------

1.9.48 (2013-08-21)
-------------------
* search for exported rosbag migration rules based on new package rosbag_migration_rule

1.9.47 (2013-07-03)
-------------------

1.9.46 (2013-06-18)
-------------------
* fix crash in bag migration (`#239 <https://github.com/ros/ros_comm/issues/239>`_)

1.9.45 (2013-06-06)
-------------------
* added option '--duration' to 'rosbag play' (`#121 <https://github.com/ros/ros_comm/issues/121>`_)
* fix missing newlines in rosbag error messages (`#237 <https://github.com/ros/ros_comm/issues/237>`_)
* fix flushing for tools like 'rosbag compress' (`#237 <https://github.com/ros/ros_comm/issues/237>`_)

1.9.44 (2013-03-21)
-------------------
* fix various issues on Windows (`#189 <https://github.com/ros/ros_comm/issues/189>`_)

1.9.43 (2013-03-13)
-------------------

1.9.42 (2013-03-08)
-------------------
* added option '--duration' to 'rosrun rosbag play' (`#121 <https://github.com/ros/ros_comm/issues/121>`_)
* add error message to rosbag when using same in and out file (`#171 <https://github.com/ros/ros_comm/issues/171>`_)

1.9.41 (2013-01-24)
-------------------

1.9.40 (2013-01-13)
-------------------
* fix bagsort script (`#42 <https://github.com/ros/ros_comm/issues/42>`_)

1.9.39 (2012-12-29)
-------------------
* first public release for Groovy
