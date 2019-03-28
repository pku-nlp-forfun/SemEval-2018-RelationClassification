import numpy as np
import os
from tqdm import tqdm

from constant import *
from util import dump_bigger, load_bigger, begin_time, end_time, loadRelation, loadPaper, loadTestEntities

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
        return load_bigger('%sdblp_embedding.pkl' % pickle_path)
    version = begin_time()
    print('Embedding pickle file not existed... Building....')
    with open(origin_embedding, 'r') as f:
        embedding_txt = f.readlines()

    embedding_pickle = {ii.split()[0]: np.array(
        ii.split()[1:]).astype(np.float) for ii in tqdm(embedding_txt)}
    embedding_mean = np.mean(list(embedding_pickle.values()))
    embedding_pickle['UNK'] = embedding_mean

    dump_bigger(embedding_pickle, '%sdblp_embedding.pkl' % pickle_path)
    end_time(version)
    return embedding_pickle


def combineWithRelationship(entity1_str: str , entity2_str: str , relation_id: int):
    """ combine entity with relation and return """
    sentences = []
    for connection in entity_relation[relation_id]:
        sentences.append(f"{entity1_str} {connection} {entity2_str}")
    return sentences


def sentencesToEmbedding(sentences: list, embedding_dict: dict):
    prod = np.ones((embedding_dim, ))
    for sentence in sentences:
        for string in sentence.split():
            try:
                prod *= embedding_dict[string]
            except KeyError:
                # Out of vocabulary
                prod *= embedding_dict['UNK']
    return prod


def stringListToEmbedding(string_list: list, embedding_dict: dict):
    # Transform sentences to embedding matrix
    sentence_embedding_product = []
    for case in string_list:
        prod = sentencesToEmbedding(case, embedding_dict)
        sentence_embedding_product.append(prod)
    
    return sentence_embedding_product


def getTrainData(embedding_dict):
    entity_pair_relations = loadRelation('%s1.1.relations.txt' % train_data_path)
    papers = loadPaper('%s1.1.text.xml' % train_data_path)

    string_list = []
    label_list = []

    for (entity1, entity2), relation in entity_pair_relations.items():
        paper1_id, entity1_id = entity1.split('.')
        paper2_id, entity2_id = entity2.split('.')
        entity1_str = papers[paper1_id].entity_str[int(entity1_id)-1] # -1 because xml
        entity2_str = papers[paper2_id].entity_str[int(entity2_id)-1] # id starts from 0
        # Form entity pair to sentences
        string_list.append(combineWithRelationship(entity1_str, entity2_str, relation))
        label_list.append(relation)
    
    sentence_embedding_product = stringListToEmbedding(string_list, embedding_dict)

    return sentence_embedding_product, label_list


def getTestData(embedding_dict):
    entity_pair = loadTestEntities('%s1.1.test.relations.txt' % test_data_path)
    papers = loadPaper('%s1.1.test.text.xml' % test_data_path)

    string_list = []

    for (entity1, entity2) in entity_pair:
        paper1_id, entity1_id = entity1.split('.')
        paper2_id, entity2_id = entity2.split('.')
        entity1_str = papers[paper1_id].entity_str[int(entity1_id)-1] # -1 because xml
        entity2_str = papers[paper2_id].entity_str[int(entity2_id)-1] # id starts from 0
        # Form entity pair to sentences
        string_list.append(f"{entity1_str} {entity2_str}")

    sentence_embedding_product = stringListToEmbedding(string_list, embedding_dict)
    return sentence_embedding_product


if __name__ == "__main__":
    # download embedding
    downloadEmbedding()
    embedding_dict = prepare_embedding()

    data, label = getTrainData(embedding_dict)
    
    print(data[:3], label[:3])
 