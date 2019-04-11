import numpy as np
import os
import pickle
import re
import random
import subprocess
import time
import urllib3
import tensorflow as tf

from constant import *
from bs4 import BeautifulSoup, NavigableString
from numba import jit

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
start = []


class Paper:
    def __init__(self, text_id: str = "", title: str = "", abstract: str = "", entity_id: list = [], entity_str: list = [], abstract_entity: str = ''):
        self.id = text_id
        self.title = title
        self.abstract = abstract
        self.abstract_entity = abstract_entity
        self.entity_id = entity_id
        self.entity_str = entity_str

    def __repr__(self):
        entity_list = ["%s: %s" % (i, string) for i, string in zip(
            self.entity_id, self.entity_str)]
        entity = "\n".join(entity_list)
        return "Paper %s\n\nTitle: %s\n\nAbstract:\n\n%s\n\nAbstract_entity:\n\n%s\n\nEntities:\n\n%s" % (self.id, self.title, self.abstract, self.abstract_entity, entity)


# i.e. Training data
def loadPaper(filename):
    papers = {}  # paper id: Paper object
    with open(filename, 'r') as text_raw:
        raw_xml = text_raw.read()

    xml = BeautifulSoup(raw_xml, features="html.parser")

    texts = xml.doc.findAll('text')

    for text in texts:
        paper = Paper(text_id=text["id"], title=text.title.text)
        abstract = text.find_all(re.compile('title|abstract'))
        paper.abstract = ' '.join([ii.text.strip() for ii in abstract])

        abstract_entity = ''

        for ii in abstract:
            for abstract_item in ii.contents:
                try:
                    abstract_entity += str(abstract_item['id'])
                except:
                    abstract_entity += str(abstract_item)
        paper.abstract_entity = abstract_entity.strip()

        entities = text.findAll('entity')
        paper.entity_id = [ii['id'] for ii in entities]
        paper.entity_str = [ii.text for ii in entities]
        papers[text["id"]] = paper

    return papers


# i.e. Training label
def loadRelation(filename):
    entity_pair_relation = {}  # (entity1, entity2): relation
    with open(filename, 'r') as relation_raw:
        raw_lines = relation_raw.readlines()
    for string in raw_lines:
        relation_str, rest_of_str = string.split('(', 1)
        relation_encode = rela2id[relation_str]
        entity1, entity2 = rest_of_str.split(',')[:2]
        if entity2.endswith(')\n'):
            entity2 = entity2[:-2]
        entity_pair_relation[(entity1, entity2)] = relation_encode

    return entity_pair_relation


# i.e. Test data
def loadTestEntities(filename):
    entity_pair = []
    with open(filename, 'r') as entities_raw:
        entity_lines = entities_raw.readlines()

    for string in entity_lines:
        rest_of_str = string.split('(', 1)[1]
        entity1, entity2 = rest_of_str.split(',')[:2]
        if entity2.endswith(')\n'):
            entity2 = entity2[:-2]

        entity_pair.append((entity1, entity2))

    return entity_pair


def formResult(test_entity: list, pred_label: list, filename: str = 'prediction.txt'):
    result = []
    for (entity1, entity2), label in zip(test_entity, pred_label):
        result.append('%s(%s,%s)' % (id2rela[label], entity1, entity2))

    with open(os.path.join(prediction_path, filename), 'w') as f:
        f.write('\n'.join(result))


def scorerEval(pred_file, key_file):
    """ evaluation using the scorer """
    cmd = f'perl {score_script_path} {pred_file} {key_file}'
    result = subprocess.check_output(
        cmd, stderr=subprocess.STDOUT, shell=True).decode()
    return result


def getMacroResult(pred_file, key_file):
    """ get macro precision, recall and f1 score """
    score = scorerEval(pred_file, key_file)
    P_str = re.findall(r'P\s+=\s+\d+\.\d+\%', score)[-1]
    P = float(re.search(r'\d+\.\d+', P_str).group(0))
    R_str = re.findall(r'R\s+=\s+\d+\.\d+\%', score)[-1]
    R = float(re.search(r'\d+\.\d+', R_str).group(0))
    F1_str = re.findall(r'F1\s+=\s+\d+\.\d+\%', score)[-1]
    F1 = float(re.search(r'\d+\.\d+', F1_str).group(0))
    return P, R, F1


@jit
def fastF1(result, predict):
    ''' f1 score '''
    true_total, r_total, p_total, p, r = 0, 0, 0, 0, 0
    total_list = []
    for trueValue in range(6):
        trueNum, recallNum, precisionNum = 0, 0, 0
        for index, values in enumerate(result):
            if values == trueValue:
                recallNum += 1
                if values == predict[index]:
                    trueNum += 1
            if predict[index] == trueValue:
                precisionNum += 1
        R = trueNum / recallNum if recallNum else 0
        P = trueNum / precisionNum if precisionNum else 0
        true_total += trueNum
        r_total += recallNum
        p_total += precisionNum
        p += P
        r += R
        f1 = (2 * P * R) / (P + R) if (P + R) else 0
        print(id2rela[trueValue], P, R, f1)
        total_list.append([P, R, f1])
    p /= 6
    r /= 6
    micro_r = true_total / r_total
    micro_p = true_total / p_total
    macro_f1 = (2 * p * r) / (p + r) if (p + r) else 0
    micro_f1 = (2 * micro_p * micro_r) / (micro_p +
                                          micro_r) if (micro_p + micro_r) else 0
    print('P: {:.2f}%, R: {:.2f}%, Micro_f1: {:.2f}%, Macro_f1: {:.2f}%'.format(
        p*100, r*100, micro_f1 * 100, macro_f1*100))
    return p, r, macro_f1, micro_f1, total_lists


def scoreSelf(predict, result=Nones):
    if result is None:
        with open('%skeys.test.1.1.txt' % test_data_path, 'r') as f:
            result = [rela2id[ii.split('(')[0]] for ii in f.readlines()]
    return fastF1(result, predict)


def begin_time():
    """
    multi-version time manage
    """
    global start
    start.append(time.time())
    return len(start) - 1


def end_time(version):
    termSpend = time.time() - start[version]
    print(str(termSpend)[0:5])


def dump_bigger(data, output_file):
    """
    pickle.dump big file which size more than 4GB
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(data, protocol=4)
    with open(output_file, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def load_bigger(input_file):
    """
    pickle.load big file which size more than 4GB
    """
    max_bytes = 2**31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(input_file)
    with open(input_file, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    return pickle.loads(bytes_in)


def load_embedding_from_txt(word2vec_model_name: str, index2word: dict):
    vocab_size = len(index2word)
    import word2vec
    word2vec_model_path = '%s%s.txt' % (embedding_dir, word2vec_model_name)
    print("using pre-trained word embedding:", word2vec_model_path)
    embed = word2vec.load(word2vec_model_path, kind='txt')
    word2vec_dict = {w: v for w, v in zip(embed.vocab, embed.vectors)}

    bound = np.sqrt(6.0) / np.sqrt(vocab_size)
    word_embedding_list = [word2vec_dict[jj] if jj in word2vec_dict else np.random.uniform(
        -bound, bound, embedding_dim) for ii, jj in index2word.items()]
    word_embedding_list[0] = np.zeros(embedding_dim)
    count_exist = len([1 for ww in index2word.values() if ww in word2vec_dict])
    count_not_exist = vocab_size - count_exist

    word_embedding_final = np.array(word_embedding_list)
    embed_dir = '%s%s%s%s.pkl' % (
        pickle_path, version, imbalance, word2vec_model_name)

    dump_bigger([word_embedding_final, count_exist,
                 count_not_exist], embed_dir)
    return word_embedding_final, count_exist, count_not_exist


def load_embedding(sess, index2word: dict, W, word2vec_model_name: str):
    ''' load embedding '''
    embed_dir = '%s%s.pkl' % (pickle_path, word2vec_model_name)
    if os.path.exists(embed_dir):
        word_embedding_final, count_exist, count_not_exist = load_bigger(
            embed_dir)
    else:
        word_embedding_final, count_exist, count_not_exist = load_embedding_from_txt(
            word2vec_model_name, index2word)

    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)
    t_assign_embedding = tf.assign(W, word_embedding)
    sess.run(t_assign_embedding)
    print("word. exists embedding:{} ;word not exist embedding: {}".format(
        count_exist, count_not_exist))
    print("using pre-trained word embedding.ended...")


# Test utilities
if __name__ == "__main__":

    P, R, F1 = getMacroResult(key_path(), key_path())
    print(P, R, F1)
    P, R, F1 = getMacroResult('%s1.1random.txt' % prediction_path, key_path())
    print(P, R, F1)

    papers = loadPaper('%s1.1.text.xml' % train_data_path)
    print(papers[0])

    relations = loadRelation('%s1.1.relations.txt' % train_data_path)
    print(list(relations)[:10])
