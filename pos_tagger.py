#!/usr/bin/env python3

# Copyright (c) 2020 Huy Ng
#
# License: MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import functools
import io
import json
import logging
import math
import os
import pickle
import signal
import sys
import time
import traceback

from pprint import pprint
from datetime import datetime
from collections import defaultdict

from numpy import save, savetxt, load, loadtxt

MY_PATH = os.path.normpath(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(MY_PATH, '.')))

from hmm import create_transition_matrix, create_emission_matrix, initialize, viterbi_forward, viterbi_backward, training_data
from pos_preproc import get_word_tag, assign_unk, processing
from syntok.tokenizer import Tokenizer

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

exit_program = False

def exit():
    global exit_program
    exit_program = True

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

CORPUS_PATH = "pos_data/WSJ_02-21.pos"
ALPHA = 0.001

TAGS= {
    'CC': ( 'conjunction, coordinating', 'and, or, but' ),
    'CD': ( 'cardinal number', 'five, three, 13%' ),
    'DT': ( 'determiner', 'the, a, these' ),
    'EX': ( 'existential there', 'there were six boys' ),
    'FW': ( 'foreign word', 'mais' ),
    'IN': ( 'conjunction, subordinating or preposition', 'of, on, before, unless' ),
    'JJ': ( 'adjective', 'nice, easy' ),
    'JJR': ( 'adjective, comparative', 'nicer, easier' ),
    'JJS': ( 'adjective, superlative', 'nicest, easiest' ),
    'LS': ( 'list item marker', ' ' ),
    'MD': ( 'verb, modal auxillary', 'may, should' ),
    'NN': ( 'noun, singular or mass', 'tiger, chair, laughter' ),
    'NNS': ( 'noun, plural', 'tigers, chairs, insects' ),
    'NNP': ( 'noun, proper singular', 'Germany, God, Alice' ),
    'NNPS': ( 'noun, proper plural', 'we met two Christmases ago' ),
    'PDT': ( 'predeterminer', 'both his children' ),
    'POS': ( 'possessive ending', '\'s' ),
    'PRP': ( 'pronoun, personal', 'me, you, it' ),
    'PRP$': ( 'pronoun, possessive', 'my, your, our' ),
    'RB': ( 'adverb', 'extremely, loudly, hard ' ),
    'RBR': ( 'adverb, comparative', 'better' ),
    'RBS': ( 'adverb, superlative', 'best' ),
    'RP': ( 'adverb, particle', 'about, off, up' ),
    'SYM': ( 'symbol', '%' ),
    'TO': ( 'infinitival to', 'what to do?' ),
    'UH': ( 'interjection', 'oh, oops, gosh' ),
    'VB': ( 'verb, base form', 'think' ),
    'VBZ': ( 'verb, 3rd person singular present', 'she thinks' ),
    'VBP': ( 'verb, non-3rd person singular present', 'I think' ),
    'VBD': ( 'verb, past tense', 'they thought' ),
    'VBN': ( 'verb, past participle', 'a sunken ship' ),
    'VBG': ( 'verb, gerund or present participle', 'thinking is fun' ),
    'WDT': ( 'wh-determiner', 'which, whatever, whichever' ),
    'WP': ( 'wh-pronoun, personal', 'what, who, whom' ),
    'WP$': ( 'wh-pronoun, possessive', 'whose, whosever' ),
    'WRB': ( 'wh-adverb', 'where, when' ),
    '.': ( 'punctuation mark, sentence closer', '.;?*' ),
    ',': ( 'punctuation mark, comma', ',' ),
    ':': ( 'punctuation mark, colon', ':' ),
    '(': ( 'contextual separator, left paren', '(' ),
    ')': ( 'contextual separator, right paren', ')' ),
}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Proper (fast) Python implementations of Dan Bernstein's DJB2 32-bit hashing function
#
# DJB2 has terrible avalanching performance, though.
# For example, it returns the same hash values for these strings: "xy", "yX", "z7".
# I recommend using Murmur3 hash. Or, at least, FNV-1a or SDBM hashes below.

djb2 = lambda x: functools.reduce(lambda x,c: 0xFFFFFFFF & (x*33 + c), x, 5381)
sdbm = lambda x: functools.reduce(lambda x,c: 0xFFFFFFFF & (x*65599 + c), x, 0)
fnv1a_32 = lambda x: functools.reduce(lambda x,c: 0xFFFFFFFF & ((x^c)*0x1000193), x, 0x811c9dc5)

assert(hex(djb2(b'hello, world'))     == '0xb0e4250d')
assert(hex(sdbm(b'hello, world'))     == '0xee6fb30c')
assert(hex(fnv1a_32(b'hello, world')) == '0x4d0ea41d')

# ...Versions for strings with regular functions

def hash_djb2(s):
    hash = 5381
    for x in s:
        hash = ((( hash << 5) + hash) + ord(x)) & 0xFFFFFFFF
    return hash

def hash_sdbm(s):
    hash = 0
    for x in s:
        hash = ((hash << 16) + (hash << 6) + ord(x) - hash) & 0xFFFFFFFF
    return hash

def hash_fnv1a_32(s):
    hash = 0x811c9dc5
    for x in s:
        hash = ((ord(x) ^ hash) * 0x01000193) & 0xFFFFFFFF
    return hash

assert(hex(hash_djb2(u'hello world, 世界')) == '0xa6bd702f')
assert(hex(hash_sdbm(u'Åland Islands'))    == '0x5f5ba0ee')
assert(hex(hash_fnv1a_32(u'Świętosław'))   == '0x16cf4524')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def build_vocab(corpus_path):
    if isinstance(corpus_path, str):
        corpus_path = [ corpus_path ]

    lines = []
    for path in corpus_path:
        logging.info(f"Reading file: '{path}'")
        with open(path, 'r') as f:
            new_lines = f.readlines()
            lines.extend(new_lines)
            logging.info(f"Data read from '{path}: {len(new_lines)} lines")

    logging.info(f"Data read in total: {len(lines)} lines")

    tokens = [line.split('\t')[0] for line in lines]
    freqs = defaultdict(int)

    for tok in tokens:
        freqs[tok] += 1

    vocab = [k for k, v in freqs.items() if (v > 1 and k != '\n')]
    unk_toks = ["--unk--", "--unk_adj--", "--unk_adv--", "--unk_digit--", "--unk_noun--", "--unk_punct--", "--unk_upper--", "--unk_verb--"]
    vocab.extend(unk_toks)
    vocab.append("--n--")
    vocab.append(" ")
    vocab = sorted(set(vocab))
    return vocab

def build_vocab2idx(corpus_path):
    vocab = build_vocab(corpus_path)
    vocab2idx = {}
    tokids = set()

    for i, tok in enumerate(sorted(vocab)):
        #~ hash32 = hash_fnv1a_32(tok)
        #~ hash16 = ((hash32 >> 16) & 0xffff) ^ (hash32 & 0xffff)
        #~ base = ((ord(tok[0]) & 0x14) << 27) + (hash16 << 11)
        #~ tokid = base
        #~ while tokid in tokids:
        #~     tokid += 1
        #~ tokids.add(tokid)

        vocab2idx[tok] = i

    return vocab2idx

def create_dictionaries(training_corpus, vocab2idx):
    emission_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    tag_counts = defaultdict(int)

    prev_tag = '--s--'

    for tok_tag in training_corpus:

        tok, tag = get_word_tag(tok_tag, vocab2idx)
        transition_counts[(prev_tag, tag)] += 1
        emission_counts[(tag, tok)] += 1
        tag_counts[tag] += 1
        prev_tag = tag

    return emission_counts, transition_counts, tag_counts

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def run_load(args):
    os.makedirs(os.path.join(MY_PATH, 'tmp'), exist_ok=True)
    logging.info("Building vocabulary index")
    vocab2idx = build_vocab2idx(CORPUS_PATH)
    logging.info(f"Saving vocabulary index ({len(vocab2idx)} elements) to 'vocab.pkl'")
    with open(os.path.join(MY_PATH, 'tmp', 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab2idx, f)
    with open(os.path.join(MY_PATH, 'tmp', 'vocab.json'), 'w') as f:
        json.dump(vocab2idx, f, indent=2)
    logging.info(f"Training the system with the data corpus in '{CORPUS_PATH}'")
    training_corpus = training_data(CORPUS_PATH)
    logging.info("Creating dictionaries")
    emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocab2idx)
    states = sorted(tag_counts.keys())
    alpha = ALPHA
    logging.info("Creating transition matrix")
    transition_matrix = create_transition_matrix(transition_counts, tag_counts, alpha)
    logging.info("Creating emission matrix")
    emission_matrix = create_emission_matrix(emission_counts, tag_counts, list(vocab2idx), alpha)
    logging.info("Saving transition matrix and emission matrix")
    save(os.path.join(MY_PATH, 'tmp', 'transition_matrix.npy'), transition_matrix)
    #~ savetxt(os.path.join(MY_PATH, 'tmp', 'transition_matrix.txt'), transition_matrix)
    save(os.path.join(MY_PATH, 'tmp', 'emission_matrix.npy'), emission_matrix)
    #~ savetxt(os.path.join(MY_PATH, 'tmp', 'emission_matrix.txt'), emission_matrix)
    return 0

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def run_predict(args):
    sample = 'My heart is always breaking for the ghosts that haunt this room.'
    #~ pprint(sample)
    #~ tokens = word_tokenize(sample)
    tok = Tokenizer()
    tokens = [token.value for token in tok.tokenize(sample)]
    #~ pprint(tokens)
    vocab2idx = build_vocab2idx(CORPUS_PATH)
    #~ file = open('vocab.pkl', 'rb')
    #~ vocab2idx = pickle.load(file)
    #~ file.close()
    #~ pprint(vocab2idx)
    
    prep_tokens = processing(vocab2idx, tokens)
    training_corpus = training_data(CORPUS_PATH)
    emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocab2idx)
    states = sorted(tag_counts.keys())
    alpha = 0.001
    transition_matrix = create_transition_matrix(transition_counts, tag_counts, alpha)
    emission_matrix = create_emission_matrix(emission_counts, tag_counts, list(vocab2idx), alpha)
    #~ transition_matrix = load('transition_matrix.npy')
    #~ emission_matrix = load('emission_matrix.npy')
    best_probs, best_paths = initialize(transition_matrix, emission_matrix, tag_counts, vocab2idx, states, prep_tokens)
    best_probs, best_paths = viterbi_forward(transition_matrix, emission_matrix, prep_tokens, best_probs, best_paths, vocab2idx)
    pred = viterbi_backward(best_probs, best_paths, states)

    res = []
    for tok, tag in zip(prep_tokens[:-1], pred[:-1]):
        res.append((tok, tag))
    for tok, tag in res:
        print(f"{tok}\t{tag}\t{TAGS[tag][0]} ({TAGS[tag][1]})")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

LOG_SIMPLE_FORMAT = "[%(pathname)s:%(lineno)d] '%(message)s'"
LOG_CONSOLE_FORMAT = "[%(pathname)s:%(lineno)d] [%(asctime)s]: '%(message)s'"
LOG_FILE_FORMAT = "[%(levelname)s] [%(pathname)s:%(lineno)d] [%(asctime)s] [%(name)s]: '%(message)s'"

LOGS_DIRECTORY = None

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

    command_group = parser.add_mutually_exclusive_group(required=True)
    command_group.add_argument("--load", help="load data",
                        action="store_const", dest="command",
                        const="load", default=None)
    command_group.add_argument("--predict", help="run prediction",
                        action="store_const", dest="command",
                        const="predict", default=None)

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
        if args.command == 'load':
            ret = run_load(args)
        if args.command == 'predict':
            ret = run_predict(args)

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
