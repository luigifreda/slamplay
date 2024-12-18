.. roscpp core prototype documentation master file, created by
   sphinx-quickstart on Fri Nov 11 11:12:23 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

A roscpp_core prototype
=======================

The goal of this prototype is to discover ways to make it easy to use
ROS message types in ways that are completely familar to the average
Linux developer.  The current prototype allows one to compile against
and serialize the most common C++ ROS messages via only a 200k debian
package.  The build dependencies are cmake, boost, and a single path.

Special thanks to Morten Kjaergaard who has been indispensible in
getting this together.


How to try it out
-----------------

Find an amd64 ubuntu machine with either ``lucid``, ``natty`` or
``oneiric`` installed.  You need root permissions.

Download the appopriate ``.deb`` file from `here
<http://people.willowgarage.com/straszheim/files>`_, e.g. on ``lucid``::

  wget http://people.willowgarage.com/straszheim/files/ros-fuerte-roscpp-core-prototype_lucid_amd64.deb

Install it::

  dpkg -i ros-fuerte-roscpp-core-prototype_lucid_amd64.deb

At this point you will see that ``/opt/ros/fuerte`` has had a bunch
of stuff dropped in to it.


To try it, create a directory src and pull down two
example files::

  % cd /tmp
  % mkdir src
  % cd src
  % wget -nv http://people.willowgarage.com/straszheim/files/CMakeLists.txt
  2011-11-11 11:52:36 URL:http://people.willowgarage.com/straszheim/files/CMakeLists.txt [268/268] -> "CMakeLists.txt" [1]
  % wget -nv http://people.willowgarage.com/straszheim/files/main.cpp
  2011-11-11 11:52:42 URL:http://people.willowgarage.com/straszheim/files/main.cpp [581/581] -> "main.cpp" [1]

the main.cpp does a very simple (de)serialization of a
``sensor_msgs/PointCloud2``::

  #include<sensor_msgs/PointCloud2.h>

  namespace rs = ros::serialization;

  int main()
  {
    sensor_msgs::PointCloud2 pc_1;
    pc_1.width = 10;
    pc_1.height = 20;
    // todo set other stuff

    std::cout << "PointCloud2 message: " << std::endl << pc_1 << std::endl;

    uint8_t buf[1024];
    rs::OStream out(buf, sizeof(buf) );
    rs::serialize(out, pc_1);

    std::cout << "Message Was Serialized" << std::endl;

    sensor_msgs::PointCloud2 pc_2;
    rs::IStream in(buf, sizeof(buf) );
    rs::deserialize(in, pc_2);

    std::cout << "Its a message again: " << std::endl << pc_2 << std::endl;
  }

and the ``CMakeLists.txt`` demonstrates the CMake incantations
required to find and use the code::

  cmake_minimum_required(VERSION 2.8.3)

  find_package(ROS 12.04 COMPONENTS
    cpp_common rostime roscpp_traits roscpp_serialization sensor_msgs)

  include_directories(${ROS_INCLUDE_DIRS})

  add_executable(something main.cpp)
  target_link_libraries(something ${ROS_LIBRARIES})

The important line is the ``find_package``.  This usage is very
standard CMake practice.  Unfortunately it is not possible to pass
``fuerte`` as the version, as cmake insists that the version be
numeric.

Indeed, four dependencies (cpp_common, rostime, roscpp_traits,
roscpp_serialization) does seem like a lot, but each is very tiny;
roscpp_traits has no library, roscpp_serialization contains only one
function.  We'll be cleaning this up.  The important bit is that the
user's workflow is decoupled from upstream.

Now build in the usual way, with the caveat that ``CMAKE_PREFIX_PATH``
must be specified for the ``find_package`` to work.  We believe that
we can remove this requirement in the future::

  % cmake . -DCMAKE_PREFIX_PATH=/opt/ros/fuerte
  -- The C compiler identification is GNU
  -- The CXX compiler identification is GNU
  -- Check for working C compiler: /usr/bin/gcc
  -- Check for working C compiler: /usr/bin/gcc -- works
  -- Detecting C compiler ABI info
  -- Detecting C compiler ABI info - done
  -- Check for working CXX compiler: /usr/bin/c++
  -- Check for working CXX compiler: /usr/bin/c++ -- works
  -- Detecting CXX compiler ABI info
  -- Detecting CXX compiler ABI info - done
  -- Configuring done
  -- Generating done
  -- Build files have been written to: /tmp/src

Make and run as usual::

  % make
  Scanning dependencies of target something
  [100%] Building CXX object CMakeFiles/something.dir/main.cpp.o
  Linking CXX executable something
  [100%] Built target something

  % ./something
  PointCloud2 message:
  header:
  [ etc ]






Contents of /opt/ros/fuerte
---------------------------

.. rubric:: bin/gen*.py

these are message generator binaries.  CMake macros provided in the
package make it easy to generate your own message types.

.. rubric:: lib/

this is currently the smallest set of libraries that you need to
serialize ROS messages.  There are three.  We will be making this
smaller, and may be able to go header-only.  Anyhow the dependencies
are fairly light::

  % ldd /opt/ros/fuerte/lib/*.so

  /opt/ros/fuerte/lib/libcpp_common.so:
    linux-vdso.so.1 =>  (0x00007fff0adc3000)
    libstdc++.so.6 => /usr/lib/libstdc++.so.6 (0x00007fbdd1bad000)
    libm.so.6 => /lib/libm.so.6 (0x00007fbdd192a000)
    libgcc_s.so.1 => /lib/libgcc_s.so.1 (0x00007fbdd1712000)
    libc.so.6 => /lib/libc.so.6 (0x00007fbdd138f000)
    /lib64/ld-linux-x86-64.so.2 (0x00007fbdd20f5000)

  /opt/ros/fuerte/lib/libroscpp_serialization.so:
    linux-vdso.so.1 =>  (0x00007fff62ff6000)
    libstdc++.so.6 => /usr/lib/libstdc++.so.6 (0x00007f7c21ae9000)
    libm.so.6 => /lib/libm.so.6 (0x00007f7c21866000)
    libgcc_s.so.1 => /lib/libgcc_s.so.1 (0x00007f7c2164e000)
    libc.so.6 => /lib/libc.so.6 (0x00007f7c212cb000)
    /lib64/ld-linux-x86-64.so.2 (0x00007f7c2202a000)

  /opt/ros/fuerte/lib/librostime.so:
    linux-vdso.so.1 =>  (0x00007fffddb60000)
    libboost_date_time.so.1.40.0 => /usr/lib/libboost_date_time.so.1.40.0 (0x00007f0af8ffa000)
    libboost_thread.so.1.40.0 => /usr/lib/libboost_thread.so.1.40.0 (0x00007f0af8de4000)
    libstdc++.so.6 => /usr/lib/libstdc++.so.6 (0x00007f0af8acf000)
    libm.so.6 => /lib/libm.so.6 (0x00007f0af884c000)
    libgcc_s.so.1 => /lib/libgcc_s.so.1 (0x00007f0af8635000)
    libc.so.6 => /lib/libc.so.6 (0x00007f0af82b1000)
    librt.so.1 => /lib/librt.so.1 (0x00007f0af80a9000)
    libpthread.so.0 => /lib/libpthread.so.0 (0x00007f0af7e8c000)
    /lib64/ld-linux-x86-64.so.2 (0x00007f0af9481000)

i.e. the only additional dependencies are ``boost::thread`` and
``boost::date_time``.  This is due to the fact that ros time classes
are considered "primtives" by ros messages.

.. rubric:: include/

the ``ROS`` headers associated with the libraries above, and generated
ROS message headers.

.. rubric:: share/msg

The original message definition (``.msg``) files.

.. rubric:: share/cmake

CMake infrastructure for straightforward finding/use of these headers.
