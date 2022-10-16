#!/usr/bin/env python3

import argparse
import json
import logging
import os
import signal
import sys
import time
import traceback

from pprint import pprint
from datetime import datetime

import aiml

MY_PATH = os.path.normpath(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(MY_PATH, '.')))

BRAIN_FILE = None
LOGS_DIRECTORY = None

exit_program = False

def exit():
    global exit_program
    exit_program = True

def run_chatbot():
    global exit_program

    k = aiml.Kernel()

    # To increase the startup speed of the bot it is
    # possible to save the parsed aiml files as a
    # dump. This code checks if a dump exists and
    # otherwise loads the aiml from the xml files
    # and saves the brain dump.
    if BRAIN_FILE is not None and os.path.exists(BRAIN_FILE):
        logging.info(f"Loading from brain file: '{BRAIN_FILE}'")
        k.loadBrain(BRAIN_FILE)
    else:
        logging.info(f"Parsing AIML files")
        k.bootstrap(learnFiles="std-startup.aiml", commands="load aiml b")
        if BRAIN_FILE is not None:
            logging.info(f"Saving brain file: '{BRAIN_FILE}'")
            k.saveBrain(BRAIN_FILE)

    # Endless loop which passes the input to the bot and prints
    # its response
    while not exit_program:
        try:
            input_text = input("> ")
        except EOFError:
            break
        response = k.respond(input_text)
        print(response)

    return 0

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

LOG_SIMPLE_FORMAT = "[%(pathname)s:%(lineno)d] '%(message)s'"
LOG_CONSOLE_FORMAT = "[%(pathname)s:%(lineno)d] [%(asctime)s]: '%(message)s'"
LOG_FILE_FORMAT = "[%(levelname)s] [%(pathname)s:%(lineno)d] [%(asctime)s] [%(name)s]: '%(message)s'"

class ColorStderr(logging.StreamHandler):
    def __init__(self, fmt=None):
        class AddColor(logging.Formatter):
            def __init__(self):
                super().__init__(fmt)
            def format(self, record: logging.LogRecord):
                msg = super().format(record)
                # Green/Cyan/Yellow/Red/Redder based on log level:
                color = '\033[1;' + ('32m', '36m', '33m', '31m', '41m')[min(4,int(4 * record.levelno / logging.FATAL))]
                return color + record.levelname + '\033[1;0m: ' + msg
        super().__init__(sys.stderr)
        self.setFormatter(AddColor())

def load_config(cfg_filename='config.json'):
    try:
        with open(os.path.join(MY_PATH, cfg_filename), 'r') as cfg:
            config = json.loads(cfg.read())
            for k in config.keys():
                if k in globals():
                    globals()[k] = config[k]
    except FileNotFoundError:
        pass

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--quiet", help="set logging to ERROR",
                        action="store_const", dest="loglevel",
                        const=logging.ERROR, default=logging.INFO)
    parser.add_argument("-d", "--debug", help="set logging to DEBUG",
                        action="store_const", dest="loglevel",
                        const=logging.DEBUG, default=logging.INFO)

    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    log_console_handler = ColorStderr(LOG_CONSOLE_FORMAT)
    log_console_handler.setLevel(args.loglevel)
    logger.addHandler(log_console_handler)

    if not LOGS_DIRECTORY is None:
        now = datetime.now()
        logs_dir = os.path.abspath(os.path.join(MY_PATH, LOGS_DIRECTORY, f"{now.strftime('%Y%m%d')}"))
        os.makedirs(logs_dir, exist_ok=True)
        log_filename = f"{now.strftime('%Y%m%d')}_{now.strftime('%H%M%S')}.txt"
        log_file_handler = logging.FileHandler(os.path.join(logs_dir, log_filename))
        log_formatter = logging.Formatter(LOG_FILE_FORMAT)
        log_file_handler.setFormatter(log_formatter)
        log_file_handler.setLevel(logging.DEBUG)
        logger.addHandler(log_file_handler)
        logging.info(f"Storing log into '{log_filename}' in '{logs_dir}'")

    ret = 0
    try:
        ret = run_chatbot()

    except Exception as e:
        logging.error(f"{type(e).__name__}: {e}")
        logging.error(traceback.format_exc())
        #~ logging.error(sys.exc_info()[2])
        ret = -1

    return ret

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
    #~ faulthandler.enable()

    #~ def sigint_handler(signum, frame):
    #~     global exit_program
    #~     logging.warning("CTRL-C was pressed")
    #~     exit_program = True
    #~     sys.exit(-2)
    #~ signal.signal(signal.SIGINT, sigint_handler)

    #~ load_config()
    sys.exit(main())
