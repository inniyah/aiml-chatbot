#!/usr/bin/env python3

"""
This file contains the PyAIML stress test.  It creates two bots, and connects
them in a cyclic loop.  A lot of output is generated; piping the results to
a log file is highly recommended.
"""

import aiml
import signal
import sys

def sigint_handler(signum, frame):
    print("CTRL-C was pressed")
    sys.exit(-2)
signal.signal(signal.SIGINT, sigint_handler)

STARTUP_AIML = "std-startup.aiml"
STARTUP_COMMAND = "load aiml b"

# Create the kernels
kern1 = aiml.Kernel()
kern1.verbose(False)
kern2 = aiml.Kernel()
kern2.verbose(False)

# Initialize the kernels
print( "Initializing Kernel #1" )
kern1.bootstrap(learnFiles=STARTUP_AIML, commands=STARTUP_COMMAND)
kern1.saveBrain("standard.brn")
print( "\nInitializing Kernel #2" )
kern2.bootstrap(brainFile="standard.brn")

# Start the bots off with some basic input.
response = "askquestion"

# Off they go!
while True:
    response = kern1.respond(response).strip()
    print( "1:", response, "\n" )
    response = kern2.respond(response).strip()
    print( "2:", response, "\n" )
    # If the robots have run out of things to say, force one of them
    # to break the ice.
    if response == "":
        response = "askquestion"
