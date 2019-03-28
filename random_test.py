# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2019-03-28 12:42:21
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-03-28 16:17:03

import random
import numpy as np
import os

from constant import *
from util import *


def load_test():
    with open('%s1.1.test.relations.txt' % test_data_path, 'r') as f:
        test_txt = f.readlines()
    predict = np.random.randint(0, 6, len(test_txt))
    result = ['%s%s' % (id2rela[predict[ii]], jj)
              for ii, jj in enumerate(test_txt)]
    with open('%s1.1random.txt' % prediction_path, 'w') as f:
        f.write(''.join(result))
    print(getMacroResult('%s1.1random.txt' % prediction_path,
                         '%skeys.test.1.1.txt' % test_data_path))

    scoreSelf(predict)


if __name__ == '__main__':
    os.makedirs(prediction_path, exist_ok=True)
    load_test()
