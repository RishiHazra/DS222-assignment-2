#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 15:25:57 2018
@author: rishi
"""

'''
Reference : https://stackoverflow.com/questions/34419645/asynchronous-computation-in-tensorflow
'''

import tensorflow as tf
import time
import numpy as np
import h5py

# cluster specification
parameter_servers = ["10.24.1.202:2220"]
workers = [ "10.24.1.207:2220", 
      "10.24.1.209:2220",
      "10.24.1.211:2220"]
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

# start a server for a specific task
server = tf.train.Server(
    cluster,
    job_name=FLAGS.job_name,
    task_index=FLAGS.task_index)

# config
batch_size = 4000
learning_rate = 0.002
training_epochs = 54
#logs_path = "/tmp/mnist/1"

if FLAGS.job_name == "ps":
    server.join()
    print('\n parameter servers initialized ...')
elif FLAGS.job_name == "worker":
    
    print('\n workers initialized ...')
    print("Data Loading ... \n")

    label_train=np.load("train_l.npy").astype(np.float32)
    label_test=np.load("test_l.npy").astype(np.float32)
    h5f1 = h5py.File('train.h5','r')
    tfidf_train = h5f1['d1'][:]
    h5f2 = h5py.File('test.h5','r')
    tfidf_test = h5f2['d2'][:]
       
    # Between-graph replication
    with tf.device(tf.train.replica_device_setter(
		worker_device="/job:worker/task:%d" % FLAGS.task_index,
		cluster=cluster)):       
        
        
        # count the number of updates
        global_step = tf.get_variable(
            'global_step',
            [],
            initializer = tf.constant_initializer(0),
			trainable = False)
        
        X = tf.placeholder(tf.float32, shape=[None, tfidf_train.shape[1]], name='image')
        Y = tf.placeholder(tf.float32, shape=[None,50],name='label')
        
        
        W = tf.get_variable(name='weights', shape=(tfidf_train.shape[1],50),
                            initializer=tf.random_normal_initializer())
        b = tf.get_variable(name='bias', shape=50, initializer= tf.random_normal_initializer())
        
        # Construct model
        logits = tf.add(tf.matmul(X,W),b)
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits= logits, name='loss')
        L = tf.reduce_mean(entropy)
        regularizer = tf.nn.l2_loss(W)
        loss = tf.reduce_mean(L + 0.01*regularizer)
        n_batches= int(tfidf_train.shape[0]/batch_size)
        
        opt = tf.train.AdamOptimizer(learning_rate= learning_rate)
        optimizer = opt.minimize(loss, global_step=global_step)
        preds = tf.nn.softmax(logits)
        #correct_preds= tf.equal(tf.argmax(preds,1), tf.argmax(Y,1))

        def accuracy(preds, labels):
            print('inside accuracy function ... \n')    
            p = 0
            for i in range(preds.shape[0]):
                if (labels[i,np.argmax(preds[i,:])]!=0):
                   p=p+1           
            return (100.0 * p/ labels.shape[0])
        
        
        init = tf.global_variables_initializer()
        print("Variables initialized ...")
        
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             global_step=global_step, init_op=init)
    
    begin_time = time.time()
    #frequency = 100
    with sv.prepare_or_wait_for_session(server.target) as sess:
        '''
        # is chief
        if FLAGS.task_index == 0:
            sv.start_queue_runners(sess, [chief_queue_runner])
            sess.run(init_token_op)
        '''
        # create log writer object (this will log on every machine)
        #writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        
        # perform training cycles
        start = time.time()
        for epoch in range(training_epochs):
            total_loss=0
            # number of batches in one epoch
            batch_count = int(tfidf_train.shape[0]/batch_size)
            
            for i in range(batch_count):
                x, y = tfidf_train[i*batch_size:(i+1)*batch_size,:],label_train[i*batch_size:(i+1)*batch_size,:]
                
                # perform the operations we defined earlier on batch
                _, l, step = sess.run([optimizer,loss,global_step],feed_dict={X:x, Y:y})
                total_loss += l
                
            print('Average loss epoch {0}: {1}'.format(epoch, total_loss/n_batches))            
        print('Total time: {0} seconds'.format(time.time()-start))
        
        # test the model
        print(' Testing phase...')
            
        print ('Train Accuracy',accuracy(sess.run(preds, feed_dict={X: tfidf_train}),label_train))
        print ('Test Accuracy {0}'.format(accuracy(sess.run(preds, feed_dict={X: tfidf_test}),label_test)))
        
        sv.stop()
    print("done")