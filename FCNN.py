from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
sys.path.append("../facenet")
sys.path.append("../facenet/src")
import os
import copy
import time
from datetime import datetime
import _pickle as pkl
import importlib
import argparse
import h5py
import matplotlib.pyplot as plt
import facenet
import facenet.src.align.detect_face
import facenet.facenet_utils as facenet_utils
import math
import random
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

class FCNN(object):
    
    def __init__(self,  args = None, model='inception_resnet_v1'):
        if args is None:
            args = self.args_maker()
        self.num_classes = args.nrof_classes
        self.input_shape = (args.image_size, args.image_size, 3)
        self.model = 'facenet.src.models.{}'.format(model)
        network = importlib.import_module('facenet.src.models.inception_resnet_v1')
        self.is_training_ph = tf.placeholder_with_default(tf.constant(False),
            shape=(), name='phase_train')
        self.in_ph = tf.placeholder(tf.float32, shape=(None, args.image_size, args.image_size, 3), name="input")
        self.out_ph = tf.placeholder(tf.int32, shape=(None), name='labels')
        self.prelogits, self.end_points = network.inference(self.in_ph, 1.0, 
                                         phase_train= self.is_training_ph, 
                                         bottleneck_layer_size=args.embedding_size,
                                         weight_decay=args.weight_decay)
        self.output = slim.fully_connected(self.prelogits, args.nrof_classes, activation_fn=None, 
        weights_initializer=slim.initializers.xavier_initializer(), 
        weights_regularizer=slim.l2_regularizer(args.weight_decay),
        scope='Logits', reuse=False)
        self.probability = tf.nn.softmax(self.output)
        correct_prediction = tf.cast(tf.equal(tf.argmax(self.output, 1), tf.cast(self.out_ph, tf.int64)), tf.float32)
        self.acc = tf.reduce_mean(correct_prediction)
        
    def load_model(self, sess, checkpoint, initialize):
        
        self.sess = sess
        pretrained_model = checkpoint
        if initialize:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
        print('Restoring pretrained model: %s' % pretrained_model)
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
        saver.restore(sess, pretrained_model)
    
    def predict(self, feed, sess=None):
        
        if sess is None:
            sess = self.sess
        if len(feed.shape)==len(self.input_shape):
            feed = np.expand_dims(feed,0)
        elif len(feed.shape)==len(self.input_shape)+1:
            feed = feed
        else:
            raise RuntimeError("Invalid Input Data Shape!")
        num = feed.shape[0]
        num_batch = int(np.ceil(num/200))
        output = np.zeros((num, self.num_classes))
        for batch in range(num_batch):
            print(batch)
            output[batch*200:(batch+1)*200] =\
            sess.run(tf.nn.softmax(self.output), {self.in_ph:feed[batch*200:(batch+1)*200]})
        return output
    
    def score(self, feed, sess=None):
        
        if sess is None:
            sess = self.sess
        if len(feed.shape)==len(self.input_shape):
            feed = np.expand_dims(feed,0)
        elif len(feed.shape)==len(self.input_shape)+1:
            feed = feed
        else:
            raise RuntimeError("Invalid Input Data Shape!")
        num = feed.shape[0]
        num_batch = int(np.ceil(num/200))
        output = np.zeros((num, self.num_classes))
        for batch in range(num_batch):
            output[batch*200:(batch+1)*200] =\
            sess.run(self.output, {self.in_ph:feed[batch*200:(batch+1)*200]})
        return output
    
    def accuracy(self, feed, sess=None):
        
        if sess is None:
            sess = self.sess
        num = feed[0].shape[0]
        num_batch = int(np.ceil(num/200))
        output = []
        for batch in range(num_batch):
            output.append(sess.run(self.acc, {self.in_ph: feed[0][batch*200:(batch+1)*200], 
                                              self.out_ph:feed[1][batch*200:(batch+1)*200]}))
        return np.mean(output)
    
    def args_maker(self):
        class argumenet_keeper(object):
            def __init__(self):
                self.nrof_classes = 2
                self.weight_decay = 0.0
                self.image_size = 160
                self.batch_size = 128
                self.max_nrof_epochs = 500
                self.epoch_size = 1000
                self.keep_probability = 1.0
                self.embedding_size = 128
                self.learning_rate = 0.0
                self.learning_rate_decay_epochs = 100
                self.learning_rate_decay_factor = 1.0
                self.center_loss_alfa = 0.95
                self.center_loss_factor = 0.01
                self.optimizer = 'ADAGRAD'
                self.saver_every = 5
                self.moving_average_decay = 0.9999
                self.log_histograms = True
                self.gpu_memory_fraction = 1.0
                self.seed = 666
                self.prelogits_norm_p = 2.0
                self.validate_every_n_epochs = 5
                self.prelogits_norm_loss_factor = 1e-4
                self.use_fixed_image_standardization = True
                self.random_rotate = True
                self.random_flip = True
                self.random_crop = True
                self.validation_set_split_ratio = 0.0
                self.filter_min_nrof_images_per_class = 0
                self.filter_percentile = 100.0
                self.filter_filename = ''
                self.learning_rate_schedule_file = 'facenet/data/learning_rate_schedule_classifier_msceleb.txt'
                self.prelogits_hist_max = 10.0
        return argumenet_keeper()