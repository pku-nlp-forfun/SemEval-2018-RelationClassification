'''
@Author: gunjianpan
@Date:   2019-04-07 21:08:47
@Last Modified by:   gunjianpan
@Last Modified time: 2019-04-10 19:24:56
'''
import tensorflow as tf
import numpy as np
import word2vec
import pickle

from text_processing import getTestData, getTrainData
from constant import *
from util import *

tf.reset_default_graph()

initializer = tf.random_normal_initializer(stddev=0.1)


def pad_middle(sent: list, max_len: int, types=1):
    ''' add padding elements (i.e. dummy word tokens) to fill the sentence to max_len '''
    entity1 = sent[0].split()
    middle_word = sent[1].split()
    entity2 = sent[2].split()
    num_pads = max_len - len(entity1) - len(entity2) - len(middle_word)
    padding = num_pads * ['[PAD]']

    if not types:  # [PAD] site
        return ' '.join([*padding, *entity1, *middle_word, *entity2])
    elif types == 1:
        return ' '.join([*entity1, *padding, *middle_word, *entity2])
    elif types == 2:
        return ' '.join([*entity1, *middle_word, *padding, *entity2])
    else:
        return ' '.join([*entity1, *middle_word, *entity2, *padding])


def main():
    train_origin_sentences = getTrainData(no_embedding=True)
    test_origin_sentences = getTestData(no_embedding=True)

    labels = [ii[-1] for ii in train_origin_sentences]

    sentences = [' '.join(ii[:-1]) for ii in train_origin_sentences]
    test_sent = [' '.join(jj[:-1]) for jj in test_origin_sentences]
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
    num_class = 6
    filter_sizes = [2, 3, 4]  # 卷积核的size
    num_filter = 3

    print(vocab_size, sent_size)

    input_x = []
    for sen in sentences_rec:
        input_x.append(np.asarray([word2index[word] for word in sen.split()]))
    output = []
    for label in labels:
        output.append(np.eye(num_class)[label])

    X = tf.placeholder(tf.int32, [None, sent_size])
    Y = tf.placeholder(tf.int32, [None, num_class])  # batch_size num_class

    Embedding = tf.Variable(tf.random_uniform(
        [vocab_size, embedding_dim], -1.0, 1.0))

    embed = tf.nn.embedding_lookup(Embedding, X)
    embed = tf.expand_dims(embed, -1)

    pool_output = []
    for i, filter_size in enumerate(filter_sizes):
        # embed batch_size sentences_size embedding_dim -1
        filter_shape = [filter_size, embedding_dim, 1, num_filter]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
        b = tf.constant(0.1, shape=[num_filter])

        conv = tf.nn.conv2d(embed,  # 卷积
                            W,
                            strides=[1, 1, 1, 1],
                            padding='VALID')
        h = tf.nn.relu(tf.nn.bias_add(conv, b))  # 非线性激活
        # max_pooling
        pool = tf.nn.max_pool(h,
                              # [batch_size, filter_height, filter_width, channel]
                              ksize=[1, sent_size-filter_size+1, 1, 1],
                              strides=[1, 1, 1, 1],
                              padding='VALID')
        pool_output.append(pool)

    filter_total = len(filter_sizes)*num_filter  # 卷积核个数*通道数
    # h_pool : [batch_size(=6), output_height(=1), output_width(=1), channel(=1) * 3]
    h_pool = tf.concat(pool_output, num_filter)

    h_pool = tf.reshape(h_pool, shape=[-1, filter_total])

    Weights = tf.get_variable('W', shape=[
        filter_total, num_class], initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.Variable(tf.constant(0.1, shape=[num_class]))
    model = tf.nn.xw_plus_b(h_pool, Weights, bias)

    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))
    optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

    prediction = tf.nn.softmax(model)
    prediction = tf.argmax(prediction, 1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        load_embedding(sess, index2word, Embedding, 'fasttext_acm')
        for i in range(5000):
            _, loss = sess.run([optimizer, cost], feed_dict={
                X: input_x, Y: output})
            if (i+1) % 1000 == 0:
                print('epoch:%d cost:%.6f' % ((i+1), loss))

        test_num = [[word2index[jj] for jj in ii.split()]
                    for ii in test_sent_out]
        predict = sess.run([prediction], feed_dict={X: test_num})[0]
        scoreSelf(predict[0])
        pickle.dump(predict, open('%spredict.pkl' % pickle_path, 'wb'))


if __name__ == '__main__':
    main()
