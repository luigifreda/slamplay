#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
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
import struct
import unittest
import time

class TestRospyExceptions(unittest.TestCase):

    def test_exceptions(self):
        # not really testing anything here other than typos
        from rospy.exceptions import ROSException, ROSSerializationException, ROSInternalException, ROSInitException, \
            TransportException, TransportTerminated, TransportInitError
        
        for e in [ROSException, ROSInitException, ROSSerializationException]:
            exc = e('foo')
            self.assert_(isinstance(exc, ROSException))
        for e in [ROSInternalException, 
                  TransportException, TransportTerminated, TransportInitError]:
            exc = e('foo')
            self.assert_(isinstance(exc, ROSInternalException))

    def test_ROSInterruptException(self):
        from rospy.exceptions import ROSInterruptException, ROSException
        try:
            raise ROSInterruptException("test")
        except ROSException:
            pass
        try:
            raise ROSInterruptException("test")
        except KeyboardInterrupt:
            pass

    def test_ROSTimeMovedBackwardsException(self):
        from rospy.exceptions import ROSTimeMovedBackwardsException, ROSInterruptException
        try:
            raise ROSTimeMovedBackwardsException(1.0)
        except ROSInterruptException as e:
            # ensure the message is not changed, because old code may check it
            self.assertEqual("ROS time moved backwards", e.message)
        try:
            time = 1.0
            raise ROSTimeMovedBackwardsException(time)
        except ROSTimeMovedBackwardsException as e:
            self.assertEqual(time, e.time)
