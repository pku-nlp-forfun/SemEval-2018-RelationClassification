import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from text_processing import getTrainData, getTestData, prepare_embedding
from constant import *
from util import scoreSelf, formResult, loadTestEntities, scorerEval


def svm(train_data, test_data, label):
    # clf = SVC()
    clf = LogisticRegression()
    clf.fit(train_data, label)

    # predict
    predict = []
    for test in test_data:
        test = np.reshape(test, (1, embedding_dim))
        predict.append(clf.predict(test))

    return predict


def lightGBM(train_data, test_data, label):
    pass


if __name__ == "__main__":

    embedding_dict = prepare_embedding()
    train_data, label = getTrainData(embedding_dict)
    test_data = getTestData(embedding_dict)
    predict = svm(train_data, test_data, label)
    print(predict[:10])
    scoreSelf(predict)

    test_entity = loadTestEntities('%skeys.test.1.1.txt' % test_data_path)
    formResult(test_entity, label, filename='svm.txt')
    print(scorerEval('%ssvm.txt' % prediction_path,
                     '%skeys.test.1.1.txt' % test_data_path))
