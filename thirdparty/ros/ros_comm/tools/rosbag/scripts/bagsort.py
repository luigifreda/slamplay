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

from __future__ import print_function

import rospy
import rosbag

def sortbags(inbag, outbag):
    rebag = rosbag.Bag(outbag, 'w')

    try:
        schedule = [(t, i) for i, (topic, msg, t) in enumerate(rosbag.Bag(inbag).read_messages(raw=True))]

        schedule = [i for (t, i) in sorted(schedule)]
        print(schedule)
    
        stage = {}
        for i, (topic, msg, t) in enumerate(rosbag.Bag(inbag).read_messages(raw=True)):
            stage[i] = (topic, msg, t)
            while (len(schedule) > 0) and (schedule[0] in stage):
                (topic, msg, t) = stage[schedule[0]]
                rebag.write(topic, msg, t, raw=True)
                del stage[schedule[0]]
                schedule = schedule[1:]

        assert schedule == []
        assert stage == {}
    finally:
        rebag.close()

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3:
        sortbags(sys.argv[1], sys.argv[2])
    else:
        print("usage: bagsort.py <inbag> <outbag>")
