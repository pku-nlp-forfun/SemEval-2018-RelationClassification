# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2019-03-28 13:01:21
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-03-28 15:55:12

test_data_path = 'data/test/'
train_data_path = 'data/'
prediction_path = 'prediction/'
score_script_path = 'data/semeval2018_task7_scorer-v1.2.pl'

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
    4: 'MODEL-FEATURE',
    5: 'COMPARE'
}
