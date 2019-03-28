import numpy as np

from sklearn.svm import SVC

from text_processing import getTrainData, getTestData, prepare_embedding
from constant import *
from util import scoreSelf

def svm(train_data, test_data, label):
    clf = SVC()
    clf.fit(train_data, label)
    
    # predict
    predict = []
    for test in test_data:
        test = np.reshape(test, (1, embedding_dim))
        predict.append(clf.predict(test))
    
    return predict


if __name__ == "__main__":

    embedding_dict = prepare_embedding()
    train_data, label = getTrainData(embedding_dict)
    test_data = getTestData(embedding_dict)
    predict = svm(train_data, test_data, label)
    print(predict[:10])
    scoreSelf(predict)
