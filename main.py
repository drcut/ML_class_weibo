#! /usr/bin/python
# -*- coding: utf8 -*-
from __future__ import print_function
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep
import numpy as np
import random
import math
import time
import os
import re
import sys
from six.moves import xrange
import layer
data_dir = "./data"                # Data directory
train_dir = os.path.join(data_dir, "/train")               # Model directory save_dir
vocab_size = 57600          #vocabulary size
_WORD_SPLIT=re.compile(b" ([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")  # regular expression for search digits
normalize_digits = True         # replace all digits to 0
model_file_name = "cnn"
# Special vocabulary symbols
_PAD = b"_PAD"                  # Padding
_GO = b"_GO"                    # start to generate the output sentence
_EOS = b"_EOS"                  # end of sentence of the output sentence
_UNK = b"_UNK"                  # unknown word
PAD_ID = 0                      # index (row number) in vocabulary
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]
#buckets = [10, 20, 30, 50,70,90,120,150,170]
buckets=[70]
steps_per_checkpoint = 100
learning_rate = 0.05
learning_rate_decay_factor = 0.99
max_gradient_norm = 2.0             # Truncated backpropagation
batch_size = 128
def read_data(source_path, target_path, buckets,max_size=None):
  """Read data from source and target files and put into buckets.
  """
  data_set = [[] for _ in buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(EOS_ID)
        for bucket_id, source_size in enumerate(buckets):
          if len(source_ids) < source_size:
            data_set[bucket_id].append([source_ids, target_ids[0]])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set
def ini_data():
    print("Prepare the raw data")
    #Create Vocabularies for both Training and Testing data.
    print()
    print("Create vocabularies")
    vocab_path = os.path.join(data_dir, "vocab.list")
    print("Vocabulary list: %s" % vocab_path)    # wmt/vocab40000.fr
    tl.nlp.create_vocabulary(vocab_path, os.path.join(data_dir,"total.txt"),
                vocab_size, tokenizer=None, normalize_digits=normalize_digits,
                _DIGIT_RE=_DIGIT_RE, _START_VOCAB=_START_VOCAB)
    #Tokenize Training and Testing data.
    print()
    print("Tokenize data")
    # normalize_digits=True means set all digits to zero, so as to reduce vocabulary size.
    tl.nlp.data_to_token_ids(os.path.join(data_dir,"raw" ,"econo.txt"), os.path.join(data_dir,"id" ,"econo_id.txt"), vocab_path,
                                tokenizer=None, normalize_digits=normalize_digits,
                                UNK_ID=UNK_ID, _DIGIT_RE=_DIGIT_RE)
    tl.nlp.data_to_token_ids(os.path.join(data_dir,"raw" ,"educate.txt"), os.path.join(data_dir,"id"  ,"educate_id.txt"), vocab_path,
                                tokenizer=None, normalize_digits=normalize_digits,
                                UNK_ID=UNK_ID, _DIGIT_RE=_DIGIT_RE)    
    tl.nlp.data_to_token_ids(os.path.join(data_dir,"raw" ,"fun.txt"), os.path.join(data_dir,"id"  ,"fun_id.txt"), vocab_path,
                                tokenizer=None, normalize_digits=normalize_digits,
                                UNK_ID=UNK_ID, _DIGIT_RE=_DIGIT_RE)
    tl.nlp.data_to_token_ids(os.path.join(data_dir,"raw" ,"health.txt"), os.path.join(data_dir,"id"  ,"health_id.txt"), vocab_path,
                                tokenizer=None, normalize_digits=normalize_digits,
                                UNK_ID=UNK_ID, _DIGIT_RE=_DIGIT_RE)
    tl.nlp.data_to_token_ids(os.path.join(data_dir,"raw" ,"house.txt"), os.path.join(data_dir,"id"  ,"house_id.txt"), vocab_path,
                                tokenizer=None, normalize_digits=normalize_digits,
                                UNK_ID=UNK_ID, _DIGIT_RE=_DIGIT_RE)
    tl.nlp.data_to_token_ids(os.path.join(data_dir,"raw" ,"military.txt"), os.path.join(data_dir,"id"  ,"military_id.txt"), vocab_path,
                                tokenizer=None, normalize_digits=normalize_digits,
                                UNK_ID=UNK_ID, _DIGIT_RE=_DIGIT_RE)
    tl.nlp.data_to_token_ids(os.path.join(data_dir,"raw" ,"money.txt"), os.path.join(data_dir ,"id" ,"money_id.txt"), vocab_path,
                                tokenizer=None, normalize_digits=normalize_digits,
                                UNK_ID=UNK_ID, _DIGIT_RE=_DIGIT_RE)
    tl.nlp.data_to_token_ids(os.path.join(data_dir,"raw" ,"science.txt"), os.path.join(data_dir,"id"  ,"science_id.txt"), vocab_path,
                                tokenizer=None, normalize_digits=normalize_digits,
                                UNK_ID=UNK_ID, _DIGIT_RE=_DIGIT_RE)
    tl.nlp.data_to_token_ids(os.path.join(data_dir,"raw" ,"sport.txt"), os.path.join(data_dir ,"id" ,"sport_id.txt"), vocab_path,
                                tokenizer=None, normalize_digits=normalize_digits,
                                UNK_ID=UNK_ID, _DIGIT_RE=_DIGIT_RE)
if __name__ == '__main__':
    ini_data()
    train_set = read_data(os.path.join(data_dir,"total_id.txt"),os.path.join(data_dir,"total_ans.txt"),buckets)

    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    print('the num of training data in each buckets: %s' % train_bucket_sizes)    # [239121, 1344322, 5239557, 10445326]
    print('the num of training data: %d' % train_total_size)        # 17268326.0
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size

for i in xrange(len(train_bucket_sizes))]
    sess = tf.InteractiveSession()
    with tf.variable_scope("model", reuse=None):
        model = layer.Cnn_Softmax_Wrapper(
                          buckets,
                          max_gradient_norm,
                          batch_size,
                          learning_rate,
                          learning_rate_decay_factor,
                          use_lstm = True,
                          forward_only=False)    # is_train = True
    # sess.run(tf.initialize_all_variables())
        layer.initialize_global_variables(sess)

        print("training")
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                               if train_buckets_scale[i] > random_number_01])

            start_time = time.time()
            encoder_inputs, decoder_inputs = model.get_batch(
                        train_set, bucket_id, PAD_ID)

            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                            bucket_id, False)
            step_time += (time.time() - start_time) / steps_per_checkpoint
            loss += step_loss / steps_per_checkpoint
            current_step += 1
        
            if current_step % steps_per_checkpoint == 0:  
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
                        "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                    step_time, perplexity))
            # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
            # Save model
                tl.files.save_npz(model.all_params, name=model_file_name+'.npz')
                step_time, loss = 0.0, 0.0
            sys.stdout.flush()
