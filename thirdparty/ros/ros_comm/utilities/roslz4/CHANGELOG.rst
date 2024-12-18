^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package roslz4
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.12.14 (2018-08-23)
--------------------

1.12.13 (2018-02-21)
--------------------
* adding decompress to free(state) before return (`#1313 <https://github.com/ros/ros_comm/issues/1313>`_)

1.12.12 (2017-11-16)
--------------------

1.12.11 (2017-11-07)
--------------------
* revert replace deprecated lz4 function call (`#1220 <https://github.com/ros/ros_comm/issues/1220>`_, regression from 1.12.8 on Debian Jessie)

1.12.10 (2017-11-06)
--------------------

1.12.9 (2017-11-06)
-------------------

1.12.8 (2017-11-06)
-------------------
* replace deprecated lz4 function call (`#1136 <https://github.com/ros/ros_comm/issues/1136>`_)

1.12.7 (2017-02-17)
-------------------

1.12.6 (2016-10-26)
-------------------

1.12.5 (2016-09-30)
-------------------

1.12.4 (2016-09-19)
-------------------

1.12.3 (2016-09-17)
-------------------
* set lz4_FOUND in order to continue using it with catkin_package(DEPENDS) (`ros/catkin#813 <https://github.com/ros/catkin/issues/813>`_)

1.12.2 (2016-06-03)
-------------------

1.12.1 (2016-04-18)
-------------------
* use directory specific compiler flags (`#785 <https://github.com/ros/ros_comm/pull/785>`_)

1.12.0 (2016-03-18)
-------------------

1.11.18 (2016-03-17)
--------------------
* fix compiler warnings

1.11.17 (2016-03-11)
--------------------

1.11.16 (2015-11-09)
--------------------

1.11.15 (2015-10-13)
--------------------

1.11.14 (2015-09-19)
--------------------

1.11.13 (2015-04-28)
--------------------

1.11.12 (2015-04-27)
--------------------

1.11.11 (2015-04-16)
--------------------
* fix import of compiled library with Python 3.x (`#563 <https://github.com/ros/ros_comm/pull/563>`_)

1.11.10 (2014-12-22)
--------------------
* disable lz4 Python bindings on Android (`#521 <https://github.com/ros/ros_comm/pull/521>`_)

1.11.9 (2014-08-18)
-------------------

1.11.8 (2014-08-04)
-------------------

1.11.7 (2014-07-18)
-------------------

1.11.6 (2014-07-10)
-------------------
* fix finding specific version of PythonLibs with CMake 3

1.11.5 (2014-06-24)
-------------------

1.11.4 (2014-06-16)
-------------------

1.11.3 (2014-05-21)
-------------------

1.11.2 (2014-05-08)
-------------------
* fix symbol problem on OSX (`#405 <https://github.com/ros/ros_comm/issues/405>`_)
* fix return value in the Python module (`#406 <https://github.com/ros/ros_comm/issues/406>`_)

1.11.1 (2014-05-07)
-------------------
* initial release
