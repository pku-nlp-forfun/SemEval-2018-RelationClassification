import numpy as np
import os
from tqdm import tqdm

from constant import *
from util import dump_bigger, load_bigger, begin_time, end_time, loadRelation, loadPaper, loadTestEntities, Paper

origin_embedding = '%sabstracts-dblp-semeval2018.wcs.txt' % pickle_path
pickle_embedding = '%sdblp_embedding.pkl' % pickle_path


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


def combineWithRelationship(entity1_str: str, entity2_str: str, relation_id: int):
    """ combine entity with relation and return """
    sentences = []
    for connection in entity_relation[relation_id]:
        sentences.append(f"{entity1_str} {connection} {entity2_str}")
    return sentences


def getMiddleWord(entity1: str, entity2: str, papers: Paper):
    ''' find middle word between entity1 and entity2 '''
    paper1_id, entity1_id = entity1.split('.')
    paper2_id, entity2_id = entity2.split('.')
    entity1_str = papers[paper1_id].entity_str[int(entity1_id)-1]
    entity2_str = papers[paper2_id].entity_str[int(entity2_id)-1]
    abstract_entity = papers[paper1_id].abstract_entity
    try:
        begin_index = abstract_entity.index(entity1)
    except:
        print(entity1, entity2, papers[paper1_id])
    end_index = abstract_entity.index(entity2)
    middle_word = abstract_entity[begin_index + len(entity1):end_index].strip()
    for ii in papers[paper1_id].entity_id:
        entity_id = int(ii.split('.')[1]) - 1
        middle_word = middle_word.replace(
            ii, papers[paper1_id].entity_str[entity_id])
    return entity1_str, middle_word, entity2_str


def sentencesToEmbedding(sentences: list, embedding_dict: dict):
    prod = np.ones((embedding_dim, ))
    for sentence in sentences:
        for string in sentence.split():
            try:
                prod *= embedding_dict[string]
            except KeyError:
                try:
                    # Look for lower case first
                    prod *= embedding_dict[string.lower()]
                except KeyError:
                    # Out of vocabulary
                    prod *= embedding_dict['UNK']
    return prod/np.linalg.norm(prod, ord=2)  # average


def stringListToEmbedding(string_list: list, embedding_dict: dict):
    # Transform sentences to embedding matrix
    sentence_embedding_product = []
    for case in string_list:
        prod = sentencesToEmbedding(case, embedding_dict)
        sentence_embedding_product.append(prod)

    return sentence_embedding_product


def getTrainData(embedding_dict={}, no_embedding=False):
    entity_pair_relations = loadRelation(train_data_txt)
    papers = loadPaper(train_data_xml)

    string_list = []
    sentences = []
    label_list = []

    for (entity1, entity2), relation in entity_pair_relations.items():
        entity1_str, middle_word, entity2_str = getMiddleWord(
            entity1, entity2, papers)
        # Form entity pair to sentences
        string_list.append(combineWithRelationship(
            entity1_str, entity2_str, relation))
        sentences.append([entity1_str, middle_word, entity2_str, relation])
        label_list.append(relation)
    if no_embedding:
        return sentences
    sentence_embedding_product = stringListToEmbedding(
        string_list, embedding_dict)

    return sentence_embedding_product, label_list


def getTestData(embedding_dict={}, no_embedding=False):
    entity_pair = loadTestEntities(test_data_txt)
    papers = loadPaper(test_data_xml)
    relation = loadRelation(test_data_key)

    string_list = []
    sentences = []

    for (entity1, entity2) in entity_pair:
        entity1_str, middle_word, entity2_str = getMiddleWord(
            entity1, entity2, papers)
        relation_id = relation[(entity1, entity2)]
        sentences.append([entity1_str, middle_word, entity2_str, relation_id])
        # Form entity pair to sentences
        string_list.append(f"{entity1_str} {entity2_str}")
    if no_embedding:
        return sentences

    sentence_embedding_product = stringListToEmbedding(
        string_list, embedding_dict)
    return sentence_embedding_product


if __name__ == "__main__":
    # download embedding
    downloadEmbedding()
    embedding_dict = prepare_embedding()

    data, label = getTrainData(embedding_dict)

    print(data[:3], label[:3])
