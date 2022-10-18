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

import string

# punctuation characters
punct = set(string.punctuation)

# morphology rules used to assign unknown word tokens
NOUN_SUFFIX = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
VERB_SUFFIX = ["ate", "ify", "ise", "ize"]
ADJ_SUFFIX  = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
ADV_SUFFIX  = ["ward", "wards", "wise"]

def get_word_tag(line, vocab):
    # check if a line is empty (just contains \n or \t), if yes
    if not line.split():
        word = "--n--"
        tag = "--s--"
        return word, tag
    else:
        word, tag = line.split()
        if word not in vocab:
            word = assign_unk(word)
        return word, tag
    return None

def preprocess(vocab, tokens):
    orig = []
    prep = []
    for cnt, word in enumerate(tokens):
        if not word.split():
            orig.append(word.strip())
            word = "--n--"
            prep.append(word)
            continue

        elif word.strip() not in vocab:
            orig.append(word.strip())
            word = assign_unk(word)
            prep.append(word)
            continue

        else:
            orig.append(word.strip())
            prep.append(word.strip())

    assert(len(orig) == len(tokens))
    assert(len(prep) == len(tokens))

    return orig, prep


def processing(vocab, text):
    prep_sentence = []
    for word in text:
        if not word.split():
            word = "--n--"
            prep_sentence.append(word)
            continue
        elif word.strip() not in vocab:
            word = assign_unk(word)
            prep_sentence.append(word)
            continue
        else:
            prep_sentence.append(word.strip())
    assert(len(prep_sentence) == len(text))
    return prep_sentence

def assign_unk(tok):
    # Digits
    if any(char.isdigit() for char in tok):
        return "--unk_digit--"

    # Punctuation
    elif any(char in punct for char in tok):
        return "--unk_punct--"

    # Upper-case
    elif any(char.isupper() for char in tok):
        return "--unk_upper--"

    # Nouns
    elif any(tok.endswith(suffix) for suffix in NOUN_SUFFIX):
        return "--unk_noun--"

    # Verbs
    elif any(tok.endswith(suffix) for suffix in VERB_SUFFIX):
        return "--unk_verb--"

    # Adjectives
    elif any(tok.endswith(suffix) for suffix in ADJ_SUFFIX):
        return "--unk_adj--"

    # Adverbs
    elif any(tok.endswith(suffix) for suffix in ADV_SUFFIX):
        return "--unk_adv--"

    return "--unk--"
