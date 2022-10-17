import time
import os.path
import unittest

from aiml import Kernel

class TestPlugins( unittest.TestCase ):

    longMessage = True

    def setUp(self):
        self.k = Kernel()
        testfile = os.path.join(os.path.dirname(__file__), "self-test.aiml")
        self.k.bootstrap(learnFiles=testfile)
        self.k.loadPlugins(os.path.join(os.path.dirname(__file__), "plugins"))

    def tearDown(self):
        del self.k

    def _testTag(self, tag, input_, outputList):
        """Tests 'tag' by feeding the Kernel 'input'.  If the result
        matches any of the strings in 'outputList', the test passes.

        """
        print( "Testing <" + tag + ">", end='\n' )
        # Send the input and collect the output
        response = self.k._cod.dec( self.k.respond( self.k._cod.enc(input_) ) )
        # Check that output is as expected
        self.assertIn( response, outputList, msg="input=%s"%input_ )

    def test01_plugin( self ):
        self._testTag('plugin', 'who is the real iron man', ["Tony Stark is the Real Iron Man!!!"])
