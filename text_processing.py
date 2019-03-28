import numpy as np
import os

from constant import *
from util import dump_bigger, begin_time, end_time

origin_embedding ='%sabstracts-dblp-semeval2018.wcs.txt' % pickle_path
pickle_embedding ='%sdblp_embedding.pkl' % pickle_path 

def downloadEmbedding():
    """ download the embedding from trenslow/LightRel """
    os.makedirs(pickle_path, exist_ok=True)
    if not os.path.isfile(origin_embedding):
        os.system('bash downloadEmbedding.sh {}'.format(origin_embedding))
    print("Embedding downloaded!")


def prepare_embedding():
    if os.path.isfile(pickle_embedding):
      print('Embedding pickle file have existed.')
      return 
    version = begin_time()
    print('Embedding pickle file not existed... Building....')
    with open(origin_embedding, 'r') as f:
        embedding_txt = f.readlines()
    embedding_pickle = {ii.split()[0]: np.array(
        ii.split()[1:]).astype(np.float) for ii in embedding_txt}

    dump_bigger(embedding_pickle, '%sdblp_embedding.pkl' % pickle_path)
    end_time(version)
    return embedding_pickle


if __name__ == "__main__":
    # download embedding
    downloadEmbedding()
    prepare_embedding()
