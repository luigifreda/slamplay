#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2009, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import os
import sys 
import unittest
import time
        
from subprocess import Popen, PIPE, check_call, call

import rospkg

def get_test_path():
    return os.path.abspath(os.path.dirname(__file__))
def get_roswtf_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class TestRoswtfOffline(unittest.TestCase):

    def setUp(self):
        pass

    ## test that the rosmsg command works
    def test_cmd_help(self):
        cmd = 'roswtf'
        output = Popen([cmd, '-h'], stdout=PIPE).communicate()[0]
        output = output.decode()
        self.assert_('Options' in output)
            
    def test_offline(self):
        cmd = 'roswtf'

        # point at a different 'master'
        env = os.environ.copy()
        env['ROS_MASTER_URI'] = 'http://localhost:11312'

        rospack = rospkg.RosPack()
        # add all dependencies to ros package path
        pkgs = ['roswtf',
            'rosgraph', 'roslaunch', 'roslib', 'rosnode', 'rosservice',
            'rosbag', 'rosbag_storage', 'roslz4', 'rosconsole', 'roscpp', 'rosgraph_msgs', 'roslang', 'rosmaster', 'rosmsg', 'rosout', 'rosparam', 'rospy', 'rostest', 'rostopic', 'topic_tools', 'xmlrpcpp',
            'cpp_common', 'roscpp_serialization', 'roscpp_traits', 'rostime',  # roscpp_core
            'rosbuild', 'rosclean', 'rosunit',  # ros
            'rospack', 'std_msgs', 'message_runtime', 'message_generation', 'gencpp', 'genlisp', 'genpy', 'genmsg', 'catkin',
        ]
        paths = [rospack.get_path(pkg) for pkg in pkgs]
        try:
            path = rospack.get_path('cmake_modules')
        except rospkg.ResourceNotFound:
            pass
        else:
            paths.append(path)
        try:
            path = rospack.get_path('geneus')
        except rospkg.ResourceNotFound:
            pass
        else:
            paths.append(path)
        try:
            path = rospack.get_path('gennodejs')
        except rospkg.ResourceNotFound:
            pass
        else:
            paths.append(path)
        env['ROS_PACKAGE_PATH'] = os.pathsep.join(paths)

        cwd  = get_roswtf_path()
        kwds = { 'env': env, 'stdout': PIPE, 'stderr': PIPE, 'cwd': cwd}

        # run roswtf nakedly
        output = Popen([cmd], **kwds).communicate()
        output = [o.decode() for o in output]

        # there should either be no errors or warnings or
        # there should be exactly one error about rosdep not being initialized
        self._check_output(output[0])

        # run roswtf on a simple launch file offline
        p = os.path.join(get_test_path(), 'min.launch')
        output = Popen([cmd, p], **kwds).communicate()[0]
        output = output.decode()
        self._check_output(output)

    def _check_output(self, output):
        # do both a positive and negative test
        self.assert_(
            'No errors or warnings' in output or 'Found 1 error' in output,
            'OUTPUT[%s]' % output)
        if 'No errors or warnings' in output:
            self.assert_('ERROR' not in output, 'OUTPUT[%s]' % output)
        if 'Found 1 error' in output:
            self.assert_(output.count('ERROR') == 1, 'OUTPUT[%s]' % output)
            self.assert_(
                'Error: the rosdep view is empty' not in output,
                'OUTPUT[%s]' % output)
