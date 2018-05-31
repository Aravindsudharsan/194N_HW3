# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import numpy as np
import tensorflow as tf

Py3 = sys.version_info[0] == 3

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      return f.read().replace("\n", "<eos>").split()
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
  data = _read_words(filename)
  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  dict_id_word = dict(zip(range(len(words)), words))
  return word_to_id , dict_id_word


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]

# Function to get the ID's for words in sentences
def _sentence_ids(filename,word_to_id,number_of_sentences):
  # reading data from text files
  data = _read_words(filename)
  # initializing variables as list
  words_sentence_output = []
  for j in range(number_of_sentences):
  	words_sentence_output.append([]) 
  print("words_sentence_output = ",words_sentence_output)
  ids_for_sentence = []
  # iterator for running
  count_sentence_no = 0
  count_word_no = 0
  # running for first 10 samples
  while(count_sentence_no < 10):
    # appending sentences from data to sentence words
    words_sentence_output[count_sentence_no].append(data[count_word_no])
    if(data[count_word_no] == '<eos>'):
      # getting sentence ID's from sentence words
    	ids_for_sentence.append([word_to_id[k] for k in words_sentence_output[count_sentence_no]])
    	count_sentence_no = count_sentence_no + 1
    count_word_no = count_word_no + 1
  # priting the ID's  
  for i in range(number_of_sentences):
    print("Sentence:",i)
    print("Sentence words are:",words_sentence_output[i])
    print("The corresponding ID's are:",ids_for_sentence[i])

# finding ID for words
def ptb_word_to_id(data, dict_word_id):
  return [dict_word_id[words] for words in data]

# getting words list
def ptb_id_to_word(data, dict_id_word):
  wordlist = []
  for words in data:
    wordlist.append([dict_id_word[wid] for wid in words])
  return wordlist

def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id , dict_id_word = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  _sentence_ids(valid_path,word_to_id,10)
  return train_data, valid_data, test_data, vocabulary , dict_id_word


def ptb_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw PTB data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    return x, y

data1 = ptb_raw_data("/Users/aravindsudharsan/Desktop/UCSB/Spring 2018/Introduction to Deep Learning/Homeworks/HW3Sudharsan_Aravind/RNN/Code/simple-examples/data")
train_data, valid_data, test_data, vocabulary, dict_id_word = data1
# print ID for all the 10,000 words
#print(dict_id_word)