^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package rospy
^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.12.14 (2018-08-23)
--------------------
* fix some errors in some probably not frequented code paths (`#1415 <https://github.com/ros/ros_comm/issues/1415>`_)
* fix thread problem with get_topics() (`#1416 <https://github.com/ros/ros_comm/issues/1416>`_)

1.12.13 (2018-02-21)
--------------------
* raise the correct exception from AnyMsg.serialize (`#1311 <https://github.com/ros/ros_comm/issues/1311>`_)

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
* change rospy.Rate hz type from int to float (`#1177 <https://github.com/ros/ros_comm/issues/1177>`_)
* improve rospy.logXXX_throttle performance (`#1091 <https://github.com/ros/ros_comm/pull/1091>`_)
* add option to reset timer when time moved backwards (`#1083 <https://github.com/ros/ros_comm/issues/1083>`_)
* abort topic lookup on connection refused (`#1044 <https://github.com/ros/ros_comm/pull/1044>`_)
* sleep in rospy wait_for_service even if exceptions raised (`#1025 <https://github.com/ros/ros_comm/pull/1025>`_)

1.12.7 (2017-02-17)
-------------------
* make get_published_topics threadsafe (`#958 <https://github.com/ros/ros_comm/issues/958>`_)
* use poll in write_header() if available to support higher numbered fileno (`#929 <https://github.com/ros/ros_comm/pull/929>`_)
* use epoll instead of poll if available to gracefully close hung connections (`#831 <https://github.com/ros/ros_comm/issues/831>`_)
* fix Python 3 compatibility issues (`#565 <https://github.com/ros/ros_comm/issues/565>`_)

1.12.6 (2016-10-26)
-------------------
* improve reconnection logic on timeout and other common errors (`#851 <https://github.com/ros/ros_comm/pull/851>`_)
* remove duplicated function (`#783 <https://github.com/ros/ros_comm/pull/783>`_)

1.12.5 (2016-09-30)
-------------------

1.12.4 (2016-09-19)
-------------------

1.12.3 (2016-09-17)
-------------------
* raise error on rospy.init_node with None or empty node name string (`#895 <https://github.com/ros/ros_comm/pull/895>`_)
* fix wrong type in docstring for rospy.Timer (`#878 <https://github.com/ros/ros_comm/pull/878>`_)
* fix order of init and publisher in example (`#873 <https://github.com/ros/ros_comm/pull/873>`_)

1.12.2 (2016-06-03)
-------------------
* add logXXX_throttle functions (`#812 <https://github.com/ros/ros_comm/pull/812>`_)

1.12.1 (2016-04-18)
-------------------

1.12.0 (2016-03-18)
-------------------

1.11.18 (2016-03-17)
--------------------

1.11.17 (2016-03-11)
--------------------
* preserve identity of numpy_msg(T) (`#758 <https://github.com/ros/ros_comm/pull/758>`_)

1.11.16 (2015-11-09)
--------------------
* catch ROSInterruptException from rospy timers when shutting down (`#690 <https://github.com/ros/ros_comm/pull/690>`_)

1.11.15 (2015-10-13)
--------------------
* validate name after remapping (`#669 <https://github.com/ros/ros_comm/pull/669>`_)

1.11.14 (2015-09-19)
--------------------
* fix memory/thread leak with QueuedConnection (`#661 <https://github.com/ros/ros_comm/pull/661>`_)
* fix signaling already shutdown to client hooks with the appropriate signature (`#651 <https://github.com/ros/ros_comm/issues/651>`_)
* fix bug with missing current logger levels (`#631 <https://github.com/ros/ros_comm/pull/631>`_)

1.11.13 (2015-04-28)
--------------------

1.11.12 (2015-04-27)
--------------------

1.11.11 (2015-04-16)
--------------------
* add rosconsole command line tool to change logger levels (`#576 <https://github.com/ros/ros_comm/pull/576>`_)
* add accessor for remaining time of the Rate class (`#588 <https://github.com/ros/ros_comm/pull/588>`_)
* fix high latency when using asynchronous publishing (`#547 <https://github.com/ros/ros_comm/issues/547>`_)
* fix error handling when publishing on Empty topic (`#566 <https://github.com/ros/ros_comm/pull/566>`_)

1.11.10 (2014-12-22)
--------------------
* add specific exception for time jumping backwards (`#485 <https://github.com/ros/ros_comm/issues/485>`_)
* make param functions thread-safe (`#523 <https://github.com/ros/ros_comm/pull/523>`_)
* fix infinitely retrying subscriber (`#533 <https://github.com/ros/ros_comm/issues/533>`_)
* fix removal of QueuedConnection leading to wrong subscriber count (`#526 <https://github.com/ros/ros_comm/issues/526>`_)
* fix TCPROS header validation when `callerid` header is not set (`#522 <https://github.com/ros/ros_comm/issues/522>`_, regression from 1.11.1)
* fix memory leak when using subcriber statistics (`#520 <https://github.com/ros/ros_comm/issues/520>`_)
* fix reported traffic in bytes from Python nodes (`#501 <https://github.com/ros/ros_comm/issues/501>`_)

1.11.9 (2014-08-18)
-------------------
* populate delivered_msgs field of TopicStatistics message (`#486 <https://github.com/ros/ros_comm/issues/486>`_)

1.11.8 (2014-08-04)
-------------------
* fix topic/connection statistics reporting code (`#482 <https://github.com/ros/ros_comm/issues/482>`_)

1.11.7 (2014-07-18)
-------------------

1.11.6 (2014-07-10)
-------------------
* make MasterProxy thread-safe (`#459 <https://github.com/ros/ros_comm/issues/459>`_)
* check ROS_HOSTNAME for localhost / ROS_IP for 127./::1 and prevent connections from other hosts in that case (`#452 <https://github.com/ros/ros_comm/issues/452>`)_

1.11.5 (2014-06-24)
-------------------

1.11.4 (2014-06-16)
-------------------
* Python 3 compatibility (`#426 <https://github.com/ros/ros_comm/issues/426>`_)

1.11.3 (2014-05-21)
-------------------
* allow shutdown hooks to be any callable object (`#410 <https://github.com/ros/ros_comm/issues/410>`_)
* add demux program and related scripts (`#407 <https://github.com/ros/ros_comm/issues/407>`_)
* add publisher queue_size to rostopic

1.11.2 (2014-05-08)
-------------------
* use publisher queue_size for statistics (`#398 <https://github.com/ros/ros_comm/issues/398>`_)

1.11.1 (2014-05-07)
-------------------
* improve asynchonous publishing performance (`#373 <https://github.com/ros/ros_comm/issues/373>`_)
* add warning when queue_size is omitted for rospy publisher (`#346 <https://github.com/ros/ros_comm/issues/346>`_)
* add optional topic/connection statistics (`#398 <https://github.com/ros/ros_comm/issues/398>`_)
* add transport information in SlaveAPI::getBusInfo() for roscpp & rospy (`#328 <https://github.com/ros/ros_comm/issues/328>`_)
* allow custom error handlers for services (`#375 <https://github.com/ros/ros_comm/issues/375>`_)
* add architecture_independent flag in package.xml (`#391 <https://github.com/ros/ros_comm/issues/391>`_)

1.11.0 (2014-03-04)
-------------------
* fix exception handling for queued connections (`#369 <https://github.com/ros/ros_comm/issues/369>`_)
* use catkin_install_python() to install Python scripts (`#361 <https://github.com/ros/ros_comm/issues/361>`_)

1.10.0 (2014-02-11)
-------------------

1.9.54 (2014-01-27)
-------------------

1.9.53 (2014-01-14)
-------------------

1.9.52 (2014-01-08)
-------------------

1.9.51 (2014-01-07)
-------------------
* implement optional queueing for rospy publications (`#169 <https://github.com/ros/ros_comm/issues/169>`_)
* overwrite __repr__ for rospy.Duration and Time (`ros/genpy#24 <https://github.com/ros/genpy/issues/24>`_)
* add missing dependency on roscpp

1.9.50 (2013-10-04)
-------------------
* add support for python coverage tool to work in callbacks

1.9.49 (2013-09-16)
-------------------

1.9.48 (2013-08-21)
-------------------
* make rospy nodes killable while waiting for master (`#262 <https://github.com/ros/ros_comm/issues/262>`_)

1.9.47 (2013-07-03)
-------------------

1.9.46 (2013-06-18)
-------------------

1.9.45 (2013-06-06)
-------------------
* add missing run_depend on python-yaml
* allow configuration of ports for XML RPCs and TCP ROS
* fix race condition where rospy subscribers do not connect to all publisher
* fix closing and deregistering connection when connect fails (`#128 <https://github.com/ros/ros_comm/issues/128>`_)
* fix log level of RosOutHandler (`#210 <https://github.com/ros/ros_comm/issues/210>`_)

1.9.44 (2013-03-21)
-------------------

1.9.43 (2013-03-13)
-------------------

1.9.42 (2013-03-08)
-------------------
* make dependencies on rospy optional by refactoring RosStreamHandler to rosgraph (`#179 <https://github.com/ros/ros_comm/issues/179>`_)

1.9.41 (2013-01-24)
-------------------

1.9.40 (2013-01-13)
-------------------
* add colorization for rospy log output (`#3691 <https://code.ros.org/trac/ros/ticket/3691>`_)
* fix socket polling under Windows (`#3959 <https://code.ros.org/trac/ros/ticket/3959>`_)

1.9.39 (2012-12-29)
-------------------
* first public release for Groovy
