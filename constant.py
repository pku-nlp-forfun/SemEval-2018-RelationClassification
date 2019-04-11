# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2019-03-28 13:01:21
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-03-28 16:30:20

test_data_path = 'data/test/'
train_data_path = 'data/'
prediction_path = 'prediction/'
score_script_path = 'data/semeval2018_task7_scorer-v1.2.pl'
pickle_path = 'pickle/'  # embedding.
embedding_dir = '../LightRel//SemEval18task7/feature/'

version = '1.1'
imbalance = 'train'
train_str = ''
test_str = '.test'
key_str = 'keys{}.'.format(test_str)
data_txt_path = '.relations.txt'
xml_path = '.text.xml'

common_path = '%s%s%s%s'
train_data_xml = common_path % (train_data_path, version, train_str, xml_path)
train_data_txt = common_path % (
    train_data_path, version, train_str, data_txt_path)

test_data_xml = common_path % (test_data_path, version, test_str, xml_path)
test_data_txt = common_path % (
    test_data_path, version, test_str, data_txt_path)
test_data_key = common_path % (test_data_path, key_str, version, '.txt')

embedding_dim = 300

rela2id = {
    'USAGE': 0,
    'TOPIC': 1,
    'RESULT': 2,
    'PART_WHOLE': 3,
    'MODEL-FEATURE': 4,
    'COMPARE': 5
}

id2rela = {
    0: 'USAGE',
    1: 'TOPIC',
    2: 'RESULT',
    3: 'PART_WHOLE',
    4: 'MODEL-FEATURE',  # MODEL
    5: 'COMPARE'        # COMPARISON
}

entity_relation = {
    0: ['used by', 'used for', 'applied to', 'performed on'],  # USAGE
    1: ['presents', 'of'],  # TOPIC
    2: ['affects', 'problem', 'yields'],  # RESULT
    3: ['composed of', 'extracted from', 'found in'],  # PART_WHOLE
    4: ['of an observed', 'of', 'associated to'],  # MODEL-FEATURE (MODEL)
    5: ['compared to'],  # COMPARE (COMPARISON)
}


def key_path(key=1):
    return '%skeys.test.1.%d.txt' % (test_data_path, key)
