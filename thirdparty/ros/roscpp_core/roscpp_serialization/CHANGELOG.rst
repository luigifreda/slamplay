^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package roscpp_serialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.7.2 (2020-05-28)
------------------

0.7.1 (2020-01-25)
------------------

0.7.0 (2020-01-24)
------------------
* various code cleanup (`#116 <https://github.com/ros/roscpp_core/issues/116>`_)
* Bump CMake version to avoid CMP0048 warning (`#115 <https://github.com/ros/roscpp_core/issues/115>`_)

0.6.13 (2019-10-03)
-------------------
* added cast to uint32_t in roscpp_serialization to fix -Wconversion warning (`#113 <https://github.com/ros/roscpp_core/issues/113>`_)
* roscpp_serialization: replace c-style-casts with static/reinterpret casts (`#107 <https://github.com/ros/roscpp_core/issues/107>`_)

0.6.12 (2019-03-04)
-------------------
* fix GCC8 class-memaccess in VectorSerializer (`#102 <https://github.com/ros/roscpp_core/issues/102>`_)

0.6.11 (2018-06-06)
-------------------
* replace reinterpret_cast with memcpy to avoid undefined behaviour/alignment issues (`#83 <https://github.com/ros/roscpp_core/issues/83>`_)

0.6.10 (2018-05-01)
-------------------

0.6.9 (2018-02-02)
------------------

0.6.8 (2018-01-26)
------------------

0.6.7 (2017-11-03)
------------------

0.6.6 (2017-10-25)
------------------

0.6.5 (2017-07-27)
------------------

0.6.4 (2017-06-06)
------------------

0.6.3 (2017-05-15)
------------------

0.6.2 (2017-02-14)
------------------
* fix warning when compiling with -Wpedantic (`#53 <https://github.com/ros/roscpp_core/issues/53>`_)

0.6.1 (2016-09-02)
------------------
* fix warning about unused parameter (`#51 <https://github.com/ros/roscpp_core/pull/51>`_)

0.6.0 (2016-03-17)
------------------

0.5.7 (2016-03-09)
------------------
* fix serializer bool on ARM (`#44 <https://github.com/ros/roscpp_core/pull/44>`_)

0.5.6 (2015-05-20)
------------------

0.5.5 (2014-12-22)
------------------

0.5.4 (2014-07-23)
------------------

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
* remove assignSubscriptionConnectionHeader (`#19 <https://github.com/ros/roscpp_core/issues/19>`_)

0.3.17 (2014-01-07)
-------------------
* move several client library independent parts from ros_comm into roscpp_core (`#12 <https://github.com/ros/roscpp_core/issues/12>`_)

0.3.16 (2013-07-14)
-------------------
* fix alignment in serialization on ARM (`#14 <https://github.com/ros/roscpp_core/issues/14>`_)

0.3.15 (2013-06-06)
-------------------
* fix compiler warning about unused variable (`#11 <https://github.com/ros/roscpp_core/issues/11>`_)
* fix install destination for dll's under Windows

0.3.14 (2013-03-21)
-------------------

0.3.13 (2013-03-08)
-------------------
* fix serialization on ARM

0.3.12 (2013-01-13)
-------------------

0.3.11 (2012-12-21)
-------------------
* first public release for Groovy
