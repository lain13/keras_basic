# -*- coding: utf-8 -*-
"""Utilities for text input preprocessing.
"""

import string
import sys
import warnings
import json
import itertools
import pickle

from collections import OrderedDict
from collections import defaultdict
from hashlib import md5
from typing import Callable
from typing import Dict
from typing import List

import numpy as np

class SentenceVectorizer(object):
    """Text vectorizer utility class.

    This class allows to vectorize a text corpus, by turning each
    text into either a sequence of integers (each integer being the index
    of a token in a dictionary) or into a vector where the coefficient
    for each token could be binary, based on word count, based on tf-idf...

    # Arguments
        max_tokens: the maximum number of words to keep, based
            on word frequency. Only the most common `max_tokens-2` words will
            be kept.
        output_sequence_length: the output will have its time dimension padded
            or truncated to exactly `output_sequence_length` values, resulting
            in a tensor of shape [batch_size, output_sequence_length] regardless
            of how many tokens resulted from the splitting step.
        nomalizer: text nomalizer callable object
        tokenizer: text tokenizer callable object
        By default, turning the texts into space-separated sequences of words
        unk_token: if given, it will be added to word_index and used to
            replace unknown words during text_to_sequence calls
        oov_token: if given, it will be added to word_index and used to
            replace out-of-vocabulary words during text_to_sequence calls

    `0` is a reserved index for empty word.
    `1` is a reserved index that out-of-vocabulary words during text_to_sequence calls
    `2` is a reserved index that won't be assigned to any word.
    """

    def __init__(self,
                 max_tokens: int,
                 output_sequence_length: int,
                 nomalizer: Callable=None,
                 tokenizer: Callable=None,
                 unk_token: str='[UNK]',
                 oov_token: str='[OOV]',
                 document_count=0):

        # OrderedDict[str, int]
        self.word_counts: OrderedDict = OrderedDict()
        self.output_sequence_length: int = output_sequence_length
        self.word_docs: defaultdict = defaultdict(int)
        self.nomalizer: Callable = nomalizer
        self.tokenizer: Callable = tokenizer
        self.max_tokens: int = max_tokens
        self.document_count: int = document_count
        self.unk_token: str = unk_token
        self.oov_token: str = oov_token
        self.index_docs: defaultdict = defaultdict(int)
        self.word_index: Dict[str, int] = {}
        self.index_word: Dict[int, str] = {}

    def fit_on_texts(self, texts) -> None:
        """Updates internal vocabulary based on a list of texts.

        In the case where texts contains lists,
        we assume each entry of the lists to be a token.

        Required before using `texts_to_sequences` or `texts_to_matrix`.

        # Arguments
            texts: can be a list of strings,
                a generator of strings (for memory-efficiency),
                or a list of list of strings.
        """

        if isinstance(texts, str):
            raise TypeError("expects an array of text on input, not a single string")

        for text in texts:
            if text is None:
                continue
            self.document_count += 1
            if self.nomalizer is not None:
                text = self.nomalizer(text)
            if self.tokenizer is not None:
                seq = self.tokenizer(text)
            else:
                seq = text.split()
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            for w in set(seq):
                # In how many documents each word occurs
                self.word_docs[w] += 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        sorted_voc = []
        # forcing the oov_token to index 1 if it exists
        if self.oov_token is not None:
            sorted_voc.append(self.oov_token)
        # forcing the unk_token to index 2 if it exists
        if self.unk_token is not None:
            sorted_voc.append(self.unk_token)
        sorted_voc.extend(wc[0] for wc in wcounts)

        # note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(
            zip(sorted_voc, list(range(1, len(sorted_voc) + 1))))

        self.index_word = {c: w for w, c in self.word_index.items()}

        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c

    def fit_on_sequences(self, sequences) -> None:
        """Updates internal vocabulary based on a list of sequences.

        Required before using `sequences_to_matrix`
        (if `fit_on_texts` was never called).

        # Arguments
            sequences: A list of sequence.
                A "sequence" is a list of integer word indices.
        """
        self.document_count += len(sequences)
        for seq in sequences:
            seq = set(seq)
            for i in seq:
                self.index_docs[i] += 1

    def texts_to_sequences(self, texts) -> np.ndarray:
        """Transforms each text in texts to a sequence of integers.

        Only top `max_tokens-2` most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.

        # Arguments
            texts: A list of texts (strings).

        # Returns
            A list of sequences.
        """
        if isinstance(texts, str):
            raise TypeError("expects an array of text on input, not a single string")

        text_len = len(texts)
        sequences = itertools.chain.from_iterable(self.texts_to_sequences_generator(texts))
        results = np.fromiter(sequences, dtype=np.int32, count=text_len * self.output_sequence_length)
        results.shape = (text_len, self.output_sequence_length)
        return results

    def texts_to_sequences_generator(self, texts):
        """Transforms each text in `texts` to a sequence of integers.

        Each item in texts can also be a list,
        in which case we assume each item of that list to be a token.

        Only top `max_tokens-2` most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.

        # Arguments
            texts: A list of texts (strings).

        # Yields
            Yields individual sequences.
        """
        max_tokens = self.max_tokens
        emt_toekn_index = 0
        oov_token_index = self.word_index.get(self.oov_token)
        unk_token_index = self.word_index.get(self.unk_token)
        for text in texts:
            if self.nomalizer is not None:
                text = self.nomalizer(text)
            if self.tokenizer is not None:
                seq = self.tokenizer(text)
            else:
                seq = text.split()
            vect = []
            for w, idx in itertools.zip_longest(seq, range(self.output_sequence_length)):
                if w is None:
                    vect.append(emt_toekn_index)
                    continue
                if idx is None:
                    break
                i = self.word_index.get(w)
                if i is not None:
                    if max_tokens and i >= max_tokens:
                        if oov_token_index is not None:
                            vect.append(oov_token_index)
                    else:
                        vect.append(i)
                elif self.unk_token is not None:
                    vect.append(unk_token_index)
                elif self.oov_token is not None:
                    vect.append(oov_token_index)
                else:
                    vect.append(emt_toekn_index)
            yield vect

    def sequences_to_texts(self, sequences) -> List[str]:
        """Transforms each sequence into a list of text.

        Only top `max_tokens-2` most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.

        # Arguments
            sequences: A list of sequences (list of integers).

        # Returns
            A list of texts (strings)
        """
        return list(self.sequences_to_texts_generator(sequences))

    def sequences_to_texts_generator(self, sequences):
        """Transforms each sequence in `sequences` to a list of texts(strings).

        Each sequence has to a list of integers.
        In other words, sequences should be a list of sequences

        Only top `max_tokens-2` most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.

        # Arguments
            sequences: A list of sequences.

        # Yields
            Yields individual texts.
        """
        max_tokens = self.max_tokens
        emt_toekn_index = 0
        oov_token_index = self.word_index.get(self.oov_token)
        unk_token_index = self.word_index.get(self.unk_token)
        for seq in sequences:
            vect = []
            for num in seq:
                if num == emt_toekn_index:
                    vect.append('')
                    continue

                word = self.index_word.get(num)
                if word is not None:
                    if max_tokens and num >= max_tokens:
                        if oov_token_index is not None:
                            vect.append(self.index_word[oov_token_index])
                    else:
                        vect.append(word)
                elif self.unk_token is not None:
                    vect.append(self.index_word[unk_token_index])
                elif self.oov_token is not None:
                    vect.append(self.index_word[oov_token_index])
                else:
                    vect.append('')
            yield vect

    def texts_to_matrix(self, texts, mode='binary'):
        """Convert a list of texts to a Numpy matrix.

        # Arguments
            texts: list of strings.
            mode: one of "binary", "count", "tfidf", "freq".

        # Returns
            A Numpy matrix.
        """
        sequences = self.texts_to_sequences(texts)
        return self.sequences_to_matrix(sequences, mode=mode)

    def sequences_to_matrix(self, sequences, mode='binary'):
        """Converts a list of sequences into a Numpy matrix.

        # Arguments
            sequences: list of sequences
                (a sequence is a list of integer word indices).
            mode: one of "binary", "count", "tfidf", "freq"

        # Returns
            A Numpy matrix.

        # Raises
            ValueError: In case of invalid `mode` argument,
                or if the Tokenizer requires to be fit to sample data.
        """
        if not self.max_tokens:
            if self.word_index:
                max_tokens = len(self.word_index) + 1
            else:
                raise ValueError('Specify a dimension (`max_tokens` argument), '
                                 'or fit on some text data first.')
        else:
            max_tokens = self.max_tokens

        if mode == 'tfidf' and not self.document_count:
            raise ValueError('Fit the Tokenizer on some data '
                             'before using tfidf mode.')

        x = np.zeros((len(sequences), max_tokens))
        for i, seq in enumerate(sequences):
            #if not seq:
            #    continue
            counts = defaultdict(int)
            for j in seq:
                if j >= max_tokens:
                    continue
                counts[j] += 1
            for j, c in list(counts.items()):
                if mode == 'count':
                    x[i][j] = c
                elif mode == 'freq':
                    x[i][j] = c / len(seq)
                elif mode == 'binary':
                    x[i][j] = 1
                elif mode == 'tfidf':
                    # Use weighting scheme 2 in
                    # https://en.wikipedia.org/wiki/Tf%E2%80%93idf
                    tf = 1 + np.log(c)
                    idf = np.log(1 + self.document_count /
                                 (1 + self.index_docs.get(j, 0)))
                    x[i][j] = tf * idf
                else:
                    raise ValueError('Unknown vectorization mode:', mode)
        return x

    def get_config(self):
        '''Returns the tokenizer configuration as Python dictionary.
        The word count dictionaries used by the tokenizer get serialized
        into plain JSON, so that the configuration can be read by other
        projects.

        # Returns
            A Python dictionary with the tokenizer configuration.
        '''

        return {
            'max_tokens': self.max_tokens,
            'output_sequence_length': self.output_sequence_length,
            'nomalizer': self.nomalizer,
            'tokenizer': self.tokenizer,
            'unk_token': self.unk_token,
            'oov_token': self.oov_token,
            'document_count': self.document_count,
            'word_counts': self.word_counts,
            'word_docs': self.word_docs,
            'index_docs': self.index_docs,
            'index_word': self.index_word,
            'word_index': self.word_index
        }

    def save(self, file_path):
        """save

        # Arguments
            file_path: path to save
        """
        config = self.get_config()
        pickle.dump(config, open(file_path, "wb"))

    @staticmethod
    def load_from_file(file_path):
        config = pickle.load(open(file_path, "rb"))
        vectorizer = SentenceVectorizer(
            max_tokens=config['max_tokens'],
            output_sequence_length=config['output_sequence_length'],
            nomalizer=config['nomalizer'],
            tokenizer=config['tokenizer'],
            unk_token=config['unk_token'],
            oov_token=config['oov_token'],
            document_count=config['document_count']
            )

        vectorizer.word_counts = config['word_counts']
        vectorizer.word_docs = config['word_docs']
        vectorizer.index_docs = config['index_docs']
        vectorizer.word_index = config['word_index']
        vectorizer.index_word = config['index_word']

        return vectorizer
