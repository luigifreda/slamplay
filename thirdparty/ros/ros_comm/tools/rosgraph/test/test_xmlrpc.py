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
import time
import mock

def test_XmlRpcHandler():
    from rosgraph.xmlrpc import XmlRpcHandler    
    # tripwire
    h = XmlRpcHandler()
    # noop
    h._ready('http://localhost:1234')
    
def test_XmlRpcNode():
    from rosgraph.xmlrpc import XmlRpcNode, XmlRpcHandler
    # not a very comprehensive test (yet)
    #port, handler
    tests = [
        (None, None, None),
        (8080, None, 8080),
        ('8080', None, 8080),
        (u'8080', None, 8080),
      ]
    
    for port, handler,true_port in tests:
        n = XmlRpcNode(port, handler)
        assert true_port == n.port
        assert handler == n.handler
        assert None == n.uri
        assert None == n.server
        n.set_uri('http://fake:1234')
        assert 'http://fake:1234' == n.uri

        n.start()
        start = time.time()
        while not n.uri and time.time() - start < 5.:
            time.sleep(0.00001) #poll for XMLRPC init

        assert n.uri
        n.shutdown('test case')

    # get coverage on run init
    n = XmlRpcNode(0, XmlRpcHandler())
    n._run_init()
    n.shutdown('test case')

    # mock in server in order to play with _run()
    n.server = mock.Mock()
    n.is_shutdown = False
    n._run_init = mock.Mock()
    n.server.serve_forever.side_effect = IOError(1, 'boom')
    n._run()

