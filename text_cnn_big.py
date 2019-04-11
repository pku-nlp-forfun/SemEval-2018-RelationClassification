'''
@Author: gunjianpan
@Date:   2019-04-08 20:23:20
@Last Modified by:   gunjianpan
@Last Modified time: 2019-04-11 21:02:24
'''

import tensorflow as tf
import numpy as np

from util import *
from constant import *
from text_processing import *


class TextCNN:
    ''' TextCNN: 1. embeddding layers, 2.conv layer, 3.max-pooling, 4.softmax layer.'''

    def __init__(self, filter_sizes, num_filters, num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length, vocab_size, embed_size, initializer=tf.random_normal_initializer(stddev=0.1), multi_label_flag=False, clip_gradients=5.0, decay_rate_big=0.50):
        """init all hyperparameter here"""
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.learning_rate = tf.Variable(
            learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(
            self.learning_rate, self.learning_rate * decay_rate_big)
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.initializer = initializer
        self.num_filters_total = self.num_filters * len(filter_sizes)
        self.multi_label_flag = multi_label_flag
        self.clip_gradients = clip_gradients
        self.is_training_flag = tf.placeholder(
            tf.bool, name="is_training_flag")

        self.input_x = tf.placeholder(
            tf.int32, [None, self.sequence_length], name="input_x")

        self.input_y_multilabel = tf.placeholder(
            tf.float32, [None, self.num_classes], name="input_y_multilabel")
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")
        self.iter = tf.placeholder(tf.int32)
        self.tst = tf.placeholder(tf.bool)
        self.use_multi_layer_cnn = False

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(
            self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.b1 = tf.Variable(tf.ones([self.num_filters]) / 10)
        self.b2 = tf.Variable(tf.ones([self.num_filters]) / 10)
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference()
        self.possibility = tf.nn.sigmoid(self.logits)
        self.loss_val = self.loss_multilabel() if multi_label_flag else self.loss()
        self.train_op = self.train()
        if not self.multi_label_flag:
            self.predictions = tf.argmax(
                self.logits, 1, name="predictions")
            print("self.predictions:", self.predictions)
            correct_prediction = tf.equal(
                tf.cast(self.predictions, tf.int32), self.input_y)
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32), name="Accuracy")

    def instantiate_weights(self):
        ''' init weight '''
        with tf.name_scope("embedding"):
            self.Embedding = tf.get_variable("Embedding", shape=[
                                             self.vocab_size, self.embed_size], initializer=self.initializer)
            self.W_projection = tf.get_variable("W_projection", shape=[
                                                self.num_filters_total, self.num_classes], initializer=self.initializer)
            self.b_projection = tf.get_variable(
                "b_projection", shape=[self.num_classes])

    def inference(self):
        ''' 1. embedding; 2. convolution -> BN -> BELU -> MAX_POOLING; 3. linear classifier;'''

        ''' embedding '''
        self.embed_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)
        self.sentence_embeddings_expanded = tf.expand_dims(
            self.embed_words, -1)

        ''' loop filter '''
        if self.use_multi_layer_cnn:  # this may take 50G memory.
            print("use multiple layer CNN")
            h = self.cnn_multiple_layers()
        else:  # this take small memory, less than 2G memory.
            print("use single layer CNN")
            h = self.cnn_single_layer()

        ''' logits(use linear layer)and predictions(argmax) '''
        with tf.name_scope("output"):
            logits = tf.matmul(h, self.W_projection) + self.b_projection
        return logits

    def cnn_single_layer(self):
        ''' single cnn '''
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("convolution-pooling-%s" % filter_size):
                ''' create filter '''
                filter = tf.get_variable("filter-%s" % filter_size, [
                                         filter_size, self.embed_size, 1, self.num_filters], initializer=self.initializer)
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[
                                    1, 1, 1, 1], padding="VALID", name="conv")
                conv = tf.contrib.layers.batch_norm(
                    conv, is_training=self.is_training_flag, scope='cnn_bn_')

                ''' apply no linearity '''
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])
                # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
                h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")
                ''' max pooling '''
                # [batch, height, width, channels]
                # ksize: A list of ints that has length >= 4.  The size of the window for each dimension of the input tensor.
                # strides: A list of ints that has length >= 4.  The stride of the sliding window for each dimension of the input tensor.
                # shape:[batch_size, 1, 1, num_filters].max_pool:performs the max pooling on the input.
                pooled = tf.nn.max_pool(h, ksize=[
                                        1, self.sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)
        ''' combine pooling feature '''
        # e.g. >>> x1=tf.ones([3,3]);x2=tf.ones([3,3]);x=[x1,x2]
        #          x12_0=tf.concat(x,0)---->x12_0' shape:[6,3]
        #          x12_1=tf.concat(x,1)---->x12_1' shape;[3,6]
        # shape:[batch_size, 1, 1, num_filters_total]. tf.concat=>concatenates tensors along one dimension.where num_filters_total=num_filters_1+num_filters_2+num_filters_3
        self.h_pool = tf.concat(pooled_outputs, 3)
        # shape should be:[None,num_filters_total]. here this operation has some result as tf.sequeeze().e.g. x's shape:[3,3];tf.reshape(-1,x) & (3, 3)---->(1,9)
        self.h_pool_flat = tf.reshape(
            self.h_pool, [-1, self.num_filters_total])

        ''' dropout '''
        with tf.name_scope("dropout"):
            # [None,num_filters_total]
            self.h_drop = tf.nn.dropout(
                self.h_pool_flat, keep_prob=self.dropout_keep_prob)
        h = tf.layers.dense(self.h_drop, self.num_filters_total,
                            activation=tf.nn.tanh, use_bias=True)
        return h

    def cnn_multiple_layers(self):
        ''' multi cnn '''
        pooled_outputs = []
        print("sentence_embeddings_expanded:",
              self.sentence_embeddings_expanded)
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope('cnn_multiple_layers' + "convolution-pooling-%s" % filter_size):
                ''' 1) CONV -> BN -> RELU '''
                filter = tf.get_variable("filter-%s" % filter_size, [
                                         filter_size, self.embed_size, 1, self.num_filters], initializer=self.initializer)
                # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[
                                    1, 1, 1, 1], padding="SAME", name="conv")
                conv = tf.contrib.layers.batch_norm(
                    conv, is_training=self.is_training_flag, scope='cnn1')
                print(i, "conv1:", conv)
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])
                # shape:[batch_size,sequence_length,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
                h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")

                ''' RESHAPE '''
                # shape:[batch_size,sequence_length,num_filters,1]
                h = tf.reshape(
                    h, [-1, self.sequence_length, self.num_filters, 1])

                ''' 2) CONV -> BN -> RELU '''
                filter2 = tf.get_variable("filter2-%s" % filter_size, [
                                          filter_size, self.num_filters, 1, self.num_filters], initializer=self.initializer)
                # shape:[batch_size,sequence_length-filter_size*2+2,1,num_filters]
                conv2 = tf.nn.conv2d(h, filter2, strides=[
                                     1, 1, 1, 1], padding="SAME", name="conv2")
                conv2 = tf.contrib.layers.batch_norm(
                    conv2, is_training=self.is_training_flag, scope='cnn2')
                print(i, "conv2:", conv2)
                b2 = tf.get_variable("b2-%s" % filter_size, [self.num_filters])
                # shape:[batch_size,sequence_length,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
                h = tf.nn.relu(tf.nn.bias_add(conv2, b2), "relu2")

                ''' 3) Max-pooling '''
                pooling_max = tf.squeeze(tf.nn.max_pool(h, ksize=[1, self.sequence_length, 1, 1], strides=[
                                         1, 1, 1, 1], padding='VALID', name="pool"))
                # pooling_avg=tf.squeeze(tf.reduce_mean(h,axis=1)) #[batch_size,num_filters]
                print(i, "pooling:", pooling_max)
                # pooling=tf.concat([pooling_max,pooling_avg],axis=1) #[batch_size,num_filters*2]
                # h:[batch_size,sequence_length,1,num_filters]
                pooled_outputs.append(pooling_max)
        ''' concat '''
        # [batch_size,num_filters*len(self.filter_sizes)]
        h = tf.concat(pooled_outputs, axis=1)
        print("h.concat:", h)

        with tf.name_scope("dropout"):
            # [batch_size,sequence_length - filter_size + 1,num_filters]
            h = tf.nn.dropout(h, keep_prob=self.dropout_keep_prob)
        return h  # [batch_size,sequence_length - filter_size + 1,num_filters]

    def loss_multilabel(self, l2_lambda=0.0001):
        ''' multi label loss '''
        with tf.name_scope("loss"):
            # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            # input_y:shape=(?, 1999); logits:shape=(?, 1999)
            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.input_y_multilabel, logits=self.logits)
            # losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
            # shape=(?, 1999).
            print("sigmoid_cross_entropy_with_logits.losses:", losses)
            # shape=(?,). loss for all data in the batch
            losses = tf.reduce_sum(losses, axis=1)
            # shape=().   average loss in the batch
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(
                v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss+l2_losses
        return loss

    def loss(self, l2_lambda=0.0001):
        ''' sing layer loss '''
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.input_y, logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(
                v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss+l2_losses
        return loss

    def train_old(self):
        ''' based on the loss, use SGD to update parameter '''
        learning_rate = tf.train.exponential_decay(
            self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(
            self.loss_val, global_step=self.global_step, learning_rate=learning_rate, optimizer="Adam", clip_gradients=self.clip_gradients)
        return train_op

    def train(self):
        ''' based on the loss, use SGD to update parameter '''
        learning_rate = tf.train.exponential_decay(
            self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        self.learning_rate_ = learning_rate
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss_val))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(zip(gradients, variables))
        return train_op


def test():
    num_classes = 6
    learning_rate = 0.001
    batch_size = 8
    decay_steps = 1000
    decay_rate = 0.95
    sequence_length = 5
    vocab_size = 10000
    embed_size = 300
    is_training = True
    dropout_keep_prob = 1.0  # 0.5
    filter_sizes = [2, 3, 4]
    num_filters = 128
    multi_label_flag = True

    sequence_length, vocab_size, input_x, output, test_num, index2word = load_data()
    textCNN = TextCNN(filter_sizes, num_filters, num_classes, learning_rate, batch_size, decay_steps,
                      decay_rate, sequence_length, vocab_size, embed_size, is_training, multi_label_flag=multi_label_flag)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        load_embedding(sess, index2word, vocab_size, textCNN, 'fasttext_acm')
        for i in range(500):
            loss, possibility, W_projection_value, _ = sess.run([textCNN.loss_val, textCNN.possibility, textCNN.W_projection, textCNN.train_op],
                                                                feed_dict={textCNN.input_x: input_x, textCNN.input_y_multilabel: output,
                                                                           textCNN.dropout_keep_prob: dropout_keep_prob, textCNN.tst: False})


def load_data():
    train_origin_sentences = getTrainData(no_embedding=True)
    test_origin_sentences = getTestData(no_embedding=True)

    labels = [ii[-1] for ii in train_origin_sentences]

    sentences = [' '.join(ii[:-1]) for ii in train_origin_sentences]
    test_sent = [' '.join(jj) for jj in test_origin_sentences]
    sentences_len = [len(ii.split()) for ii in [*sentences, *test_sent]]
    sent_size = max(sentences_len)

    sentences_rec = [pad_middle(ii, sent_size)
                     for ii in train_origin_sentences]
    test_sent_out = [pad_middle(ii, sent_size) for ii in test_origin_sentences]

    # result = [ii[-1] for ii in test_origin_sentences]

    wordlist = ' '.join([*sentences, *test_sent]).split()
    wordlist = sorted(list(set(wordlist)))
    wordlist = ['[PAD]', *wordlist]
    word2index = {w: i for i, w in enumerate(wordlist)}
    index2word = {i: w for w, i in word2index.items()}
    vocab_size = len(word2index)
    input_x = []
    for sen in sentences_rec:
        input_x.append(np.asarray([word2index[word] for word in sen.split()]))
    output = []
    for label in labels:
        output.append(np.eye(embedding_dim)[label])
    test_num = [[word2index[jj] for jj in ii.split()] for ii in test_sent_out]

    return sent_size, vocab_size, input_x, output, test_num, index2word
