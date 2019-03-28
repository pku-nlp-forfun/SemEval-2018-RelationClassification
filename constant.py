# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2019-03-28 13:01:21
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-03-28 16:30:20

test_data_path = 'data/test/'
train_data_path = 'data/'
prediction_path = 'prediction/'
score_script_path = 'data/semeval2018_task7_scorer-v1.2.pl'
pickle_path = 'pickle/' # embedding.

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
    4: 'MODEL-FEATURE', # MODEL
    5: 'COMPARE'        # COMPARISON
}

entity_relation = {
    0: ['used by', 'used for', 'applied to', 'performed on'], # USAGE
    1: ['presents', 'of'], # TOPIC
    2: ['affects', 'problem', 'yields'], # RESULT
    3: ['composed of', 'extracted from', 'found in'], # PART_WHOLE
    4: ['of an observed', 'of', 'associated to'], # MODEL-FEATURE (MODEL)
    5: ['compared to'], # COMPARE (COMPARISON)
}

def key_path(key=1):
    return '%skeys.test.1.%d.txt' % (test_data_path, key)
