
import tensorflow as tf
import numpy as np
from text_cnn_big import TextCNN
import pickle
import h5py
import os
import random
import pandas as pd

from sklearn.model_selection import train_test_split
from util import *
from constant import *
from text_processing import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float("learning_rate", 0.0003, "learning rate")
tf.app.flags.DEFINE_integer(
    "batch_size", 64, "Batch size for training/evaluating.")  # 批处理的大小 32-->128
tf.app.flags.DEFINE_integer(
    "decay_steps", 1000, "how many steps before decay learning rate.")  # 6000批处理的大小 32-->128
tf.app.flags.DEFINE_float(
    "decay_rate", 1.0, "Rate of decay for learning rate.")  # 0.65一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir", "checkpoint/",
                           "checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len", 200, "max sentence length")
tf.app.flags.DEFINE_integer("embed_size", 300, "embedding size")
tf.app.flags.DEFINE_boolean(
    "is_training_flag", True, "is training.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs", 1000, "number of epochs to run.")
tf.app.flags.DEFINE_integer(
    "validate_every", 10, "Validate every validate_every epochs.")  # 每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding", True,
                            "whether to use embedding or not.")
tf.app.flags.DEFINE_integer(
    "num_filters", 128, "number of filters")  # 256--->512
tf.app.flags.DEFINE_string("embedding_name",
                           "fasttext_v10", "word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_string("name_scope", "cnn", "name scope value.")
tf.app.flags.DEFINE_boolean(
    "multi_label_flag", True, "use multi label or single label.")
tf.app.flags.DEFINE_integer("pad_type", 0, "pad type")

filter_sizes = [6, 7, 8]


def main():
    ''' 1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction) '''

    word2index, label2index, trainX, trainY, vaildX, vaildY, testX, testY = load_data()
    max_f1_micro, max_f1_macro, max_p, max_r, test_f1_micro, test_f1_macro, test_p, test_r, max_label_list, test_label_list = [
        *[0.0] * 8, *[[]] * 2]

    vocab_size, num_classes = [len(word2index), len(label2index)]
    print("cnn_model.vocab_size: {}, num_classes: {}".format(
        vocab_size, num_classes))

    num_examples, FLAGS.sentence_len = trainX.shape
    print("num_examples of training:", num_examples,
          ";sentence_len:", FLAGS.sentence_len)

    ''' print some message for debug purpose '''
    print("trainX[0:10]:", trainX[0:10])
    print("trainY[0]:", trainY[0:10])
    print("train_y_short:", trainY[0])

    ''' 2.create session '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        ''' Instantiate Model '''
        textCNN = TextCNN(filter_sizes, FLAGS.num_filters, num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                          FLAGS.decay_rate, FLAGS.sentence_len, vocab_size, FLAGS.embed_size, multi_label_flag=FLAGS.multi_label_flag)
        ''' Initialize Save '''
        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint.")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding:
                index2word = {v: k for k, v in word2index.items()}
                load_embedding(sess, index2word,
                               textCNN.Embedding, FLAGS.embedding_name)
        current_epoch = sess.run(textCNN.epoch_step)

        ''' 3.feed data & training '''
        number_of_training_data = len(trainX)
        batch_size = FLAGS.batch_size
        iteration = 0
        for epoch in range(current_epoch, FLAGS.num_epochs):
            loss, counter = 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size), range(batch_size, number_of_training_data, batch_size)):
                iteration = iteration+1
                if not epoch and not counter:
                    print("trainX[start:end]:", trainX[start:end])
                feed_dict = {
                    textCNN.input_x: trainX[start:end], textCNN.dropout_keep_prob: 0.8, textCNN.is_training_flag: FLAGS.is_training_flag}
                if not FLAGS.multi_label_flag:
                    feed_dict[textCNN.input_y] = trainY[start:end]
                else:
                    feed_dict[textCNN.input_y_multilabel] = trainY[start:end]
                curr_loss, lr, _ = sess.run(
                    [textCNN.loss_val, textCNN.learning_rate, textCNN.train_op], feed_dict)
                loss, counter = loss + curr_loss, counter + 1
                if not counter % 50:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tLearning rate:%.5f" % (
                        epoch, counter, loss/float(counter), lr))

                ''' vaild model '''
                if not epoch % FLAGS.validate_every:
                    eval_loss, f1_score, f1_micro, f1_macro, p, r, label_list = do_eval(
                        sess, textCNN, vaildX, vaildY, num_classes)
                    print("Epoch %d Validation Loss:%.3f\tF1 Score:%.3f\tF1_micro:%.3f\tF1_macro:%.3f\tP:%.3f\tR:%.3f" % (
                        epoch, eval_loss, f1_score, f1_micro, f1_macro, p, r))
                    if f1_macro > max_f1_macro:
                        max_f1_macro, max_f1_micro, max_p, max_r, max_label_list = [
                            f1_macro, f1_micro, p, r, label_list]
                        test_loss, f1_score, test_f1_micro, test_f1_macro, test_p, test_r, test_label_list = do_eval(
                            sess, textCNN, testX, testY, num_classes)
                        print("Test Loss:%.3f\tF1 Score:%.3f\tF1_micro:%.3f|F1_macro:%.3f|P:%.3f|R:%.3f" % (
                            test_loss, f1_score, test_f1_micro, test_f1_macro, test_p, test_r))
                        print('|'.join(['{:.2f}'.format(ii)
                                        for ii in test_label_list]))
                    ''' save model to checkpoint '''
                    save_path = FLAGS.ckpt_dir + "model.ckpt"
                    saver.save(sess, save_path, global_step=epoch)
            sess.run(textCNN.epoch_increment)

            ''' test model '''
            if not epoch % 100:
                eval_loss, f1_score, f1_micro, f1_macro, p, r, _ = do_eval(
                    sess, textCNN, testX, testY, num_classes)
                print("Epoch %d Test Loss:%.3f\tF1 Score:%.3f\tF1_micro:%.3f\tF1_macro:%.3f\tP:%.3f\tR:%.3f" % (
                    epoch, eval_loss, f1_score, f1_micro, f1_macro, p, r))

        ''' print train best '''
        print("Train MAX F1_micro:%.3f|%.3f|%.3f|%.3f" %
              (max_f1_micro * 100, max_f1_macro * 100, max_p * 100, max_r * 100))
        print('|'.join(['{:.2f}'.format(ii * 100) for ii in max_label_list]))
        print("Test F1_micro:%.3f|%.3f|%.3f|%.3f" % (test_f1_macro *
                                                     100, test_f1_micro * 100, test_p * 100, test_r * 100))
        print('|'.join(['{:.2f}'.format(ii * 100) for ii in test_label_list]))


def do_eval(sess, textCNN, evalX, evalY, num_classes):
    evalX = evalX[0:3000]
    evalY = evalY[0:3000]
    number_examples = len(evalX)
    eval_loss, eval_counter, eval_f1_score, eval_p, eval_r = 0.0, 0, 0.0, 0.0, 0.0
    batch_size = FLAGS.batch_size
    predict = []

    for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
        ''' evaluation in one batch '''
        feed_dict = {textCNN.input_x: evalX[start:end], textCNN.input_y_multilabel: evalY[start:end], textCNN.dropout_keep_prob: 1.0,
                     textCNN.is_training_flag: False}
        current_eval_loss, logits = sess.run(
            [textCNN.loss_val, textCNN.logits], feed_dict)
        predict += logits[0]
        eval_loss += current_eval_loss
        eval_counter += 1

    if not FLAGS.multi_label_flag:
        predict = [int(ii > 0.5) for ii in predict]
    p, r, f1_macro, f1_micro, label_list = scoreSelf(predict, evalY)
    f1_score = (f1_micro+f1_macro)/2.0
    return eval_loss/float(eval_counter), f1_score, f1_micro, f1_macro, p, r, label_list


def pad_middle(sent: list, max_len: int, types=FLAGS.pad_type):
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


def load_data():
    train_origin_sentences = getTrainData(no_embedding=True)
    test_origin_sentences = getTestData(no_embedding=True)

    labels = [np.eye(6)[ii[-1]] for ii in train_origin_sentences]

    sentences = [' '.join(ii[:-1]) for ii in train_origin_sentences]
    test_sent = [' '.join(jj[:-1]) for jj in test_origin_sentences]
    sentences_len = [len(ii.split()) for ii in [*sentences, *test_sent]]
    sent_size = max(sentences_len)
    wordlist = ' '.join([*sentences, *test_sent]).split()
    wordlist = sorted(list(set(wordlist)))
    wordlist = ['[PAD]', *wordlist]
    word2index = {w: i for i, w in enumerate(wordlist)}

    sentences_rec = [pad_middle(ii, sent_size)
                     for ii in train_origin_sentences]
    sentences_rec = [np.asarray([word2index[word]
                                 for word in ii.split()]) for ii in sentences_rec]

    test_X = [pad_middle(ii, sent_size) for ii in test_origin_sentences]
    test_X = [np.asarray([word2index[word] for word in ii.split()])
              for ii in test_X]
    test_Y = [np.eye(6)[ii[-1]] for ii in test_origin_sentences]

    # result = [ii[-1] for ii in test_origin_sentences]

    label2index = {ii: ii for ii in range(6)}
    train_X, X_test, train_Y, y_test = train_test_split(
        sentences_rec, labels, test_size=0.25)
    train_X = pd.DataFrame(train_X)

    return word2index, label2index, train_X, train_Y, X_test, y_test, test_X, test_Y


if __name__ == "__main__":
    tf.app.run()
