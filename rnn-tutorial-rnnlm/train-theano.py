#! /usr/bin/env python
#encoding=utf-8

import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime
from utils import *
from rnn_theano import RNNTheano
from RnnNumpy import RNNNumpy

_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '8000'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '8'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '100'))
_MODEL_FILE = os.environ.get('MODEL_FILE')

def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=1, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
            # ADDED! Saving model oarameters
            #save_model_parameters_theano("./data/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
#             print model.U
#             print model.V
#             print model.W
        # For each training example...
        for i in range(len(y_train)): #每一个样本,一个样本是一个句子
            # One SGD step
            print " X_train[",  i , "]: ", X_train[i]
            print " X_train[",  i , "].len: ", len(X_train[i])
            print " y_train[",  i , "]: ", y_train[i]
            print " y_train[",  i , "].len: ", len(y_train[i])
            print " y_train[",  i , "].type: ", type(y_train[i])
            model.sgd_step(X_train[i], y_train[i], learning_rate) #一次使用一个样本跟新参数，即batch 大小为1
            num_examples_seen += 1

vocabulary_size = _VOCABULARY_SIZE
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# Read the data and append SENTENCE_START and SENTENCE_END tokens
print "Reading CSV file..."
# with open('data/reddit-comments-2015-08.csv', 'rb') as f:
with open('data/script.txt', 'rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print "sentences:"
    for x in sentences:
        print x
print "Parsed %d sentences." % (len(sentences))
    
# Tokenize the sentences into words
#分词,分隔符是空格
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
#for x in tokenized_sentences:
#    for y in x:
#        print  y

# Count the word frequencies
#统计每个词出现的次数
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
#for item in word_freq.items():
#    print item[0], item[1]
vocabulary_size = len(word_freq.items())
print "Found %d unique words tokens." % len(word_freq.items())

# Get the most common words and build index_to_word and word_to_index vectors
#去掉出现次数最少的一个word
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
#print index_to_word
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
print "word_to_index:"
#统计word到索引下标的映射
for w, i in word_to_index.items():
    print w, i

print "Using vocabulary size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
#    for x in tokenized_sentences[i]:
#        print x

# Create the training data
#通过word_to_index把处理过的句子变成索引矩阵
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
#注意，这个问题的y是当前word的下一个word
print "X_train:"
print X_train
print "X_train.shape:", X_train.shape
print "Y_train:"
print y_train
print "Y_train.shape:", y_train.shape
print "HIDDEN_DIM", _HIDDEN_DIM
model = RNNNumpy(vocabulary_size, hidden_dim=_HIDDEN_DIM)
# model = RNNTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM)
# t1 = time.time()
# model.sgd_step(X_train[10], y_train[10], _LEARNING_RATE)
# t2 = time.time()
# print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)

if _MODEL_FILE != None:
    load_model_parameters_theano(_MODEL_FILE, model)

train_with_sgd(model, X_train, y_train, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)
