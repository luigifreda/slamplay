^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package cpp_common
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.7.2 (2020-05-28)
------------------

0.7.1 (2020-01-25)
------------------
* only depend on the boost components needed (`#117 <https://github.com/ros/roscpp_core/issues/117>`_)

0.7.0 (2020-01-24)
------------------
* various code cleanup (`#116 <https://github.com/ros/roscpp_core/issues/116>`_)
* Bump CMake version to avoid CMP0048 warning (`#115 <https://github.com/ros/roscpp_core/issues/115>`_)

0.6.13 (2019-10-03)
-------------------

0.6.12 (2019-03-04)
-------------------
* update the use of macros in platform.h (`#99 <https://github.com/ros/roscpp_core/issues/99>`_)
* avoid unnecessary memory allocation (std::string) (`#95 <https://github.com/ros/roscpp_core/issues/95>`_)
* fix bug in HAVE_CXXABI_H compiler check (`#89 <https://github.com/ros/roscpp_core/issues/89>`_)

0.6.11 (2018-06-06)
-------------------

0.6.10 (2018-05-01)
-------------------

0.6.9 (2018-02-02)
------------------

0.6.8 (2018-01-26)
------------------
* define console_bridge macro with API call rather than relying on short macro being defined (`#71 <https://github.com/ros/roscpp_core/issues/71>`_)

0.6.7 (2017-11-03)
------------------
* fix support for console_bridge < 0.3.0 (`#70 <https://github.com/ros/roscpp_core/issues/70>`_, regression of 0.6.6)

0.6.6 (2017-10-25)
------------------
* replace usage of deprecated logError macros (`#69 <https://github.com/ros/roscpp_core/issues/69>`_)

0.6.5 (2017-07-27)
------------------

0.6.4 (2017-06-06)
------------------

0.6.3 (2017-05-15)
------------------

0.6.2 (2017-02-14)
------------------

0.6.1 (2016-09-02)
------------------

0.6.0 (2016-03-17)
------------------

0.5.7 (2016-03-09)
------------------
* export symbols for Header (`#46 <https://github.com/ros/roscpp_core/pull/46>`_)

0.5.6 (2015-05-20)
------------------

0.5.5 (2014-12-22)
------------------

0.5.4 (2014-07-23)
------------------

0.5.3 (2014-06-28)
------------------
* add missing boost dependency (`#24 <https://github.com/ros/roscpp_core/issues/24>`_)

0.5.2 (2014-06-27)
------------------
* find_package console_bridge with REQUIRED (`#23 <https://github.com/ros/roscpp_core/issues/23>`_)

0.5.1 (2014-06-24)
------------------
* convert to use console bridge from upstream debian package (`ros/rosdistro#4633 <https://github.com/ros/rosdistro/issues/4633>`_)

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
* move several client library independent parts from ros_comm into roscpp_core (`#12 <https://github.com/ros/roscpp_core/issues/12>`_)

0.3.16 (2013-07-14)
-------------------

0.3.15 (2013-06-06)
-------------------
* fix install destination for dll's under Windows

0.3.14 (2013-03-21)
-------------------

0.3.13 (2013-03-08)
-------------------

0.3.12 (2013-01-13)
-------------------

0.3.11 (2012-12-21)
-------------------
* first public release for Groovy
