import numpy as np
import tensorflow as tf

word_embedding_file = 'data/en-cw.txt' # Pre-trained word embeddings

pos_prefix = '<pos> :'
label_prefix = '<label> :'
global_root = '<root>'
global_unknown = '<unknown>'
unknown_prefix = '<unknown> :'
global_null = '<null>'

def read_word_embeddings():
    '''read pre-trained word-embeddings from data/en-cw.txt'''
    word_vectors = {}
    for line in open(word_embedding_file).readlines():
        sp = line.strip().split()
        word_vectors[sp[0]] = [float(x) for x in sp[1:]]
    return word_vectors

def load_train_data(data_type):
    '''load train data from file
    Sample line:
    ^17 \t Industrial \t _ \t PROPN \t NNP \t _ \t 18 \t compound \t _ \t _$
    ^order in sentence \t token \t _ \t POS tag of Stanford \t POS tag of Penn treebank \t _ \t order of token which current token depend to \t label \t _ \t _$
    '''
    train_data = [] # empty list for storing all sentence
    with open ('./data/'+ data_type + '.conll') as file: # open file
        data = [] # empty list for store element in current sentence
        for line in file:
            es = line.split('\t') # split line with delimiter '\t'
            if len(es) >= 8: # if number of elements greater than or equal to 8, valid line
                data.append(es)
            else:
                train_data.append(data) # encounter an empty line. A sentence finish add to train_data
                data = []
    return train_data

def prepare_data(data_type):
    '''prepare train data for parsing'''
    train_data = load_train_data(data_type) # read train data according to type ('dev' 'test')
    word2vector = read_word_embeddings() # read pre-trained word embeddings

    unknown_words_dict = {} # empty dict for storing words which not in pre-trained word embeddings
    pos_dict = {} # empty for storing all appeared POS tag in the train data
    label_dict = {} # empty for storing all appeared Lable in the train data
    for dependencies in train_data: # every element in train data is all dependencies in a sentence
        for dependency in dependencies: # every dependency of a sentence
            dpc = dependency
            pos_dict[dpc[4]] = 0 # push penn treebank POS tag in pos_dict
            label_dict[dpc[7]] = 0 # push label in label_dict
            if dpc[1] not in word2vector and dpc[1].lower() not in word2vector:
                unknown_words_dict[dpc[4] + ':'] = 0 # for unknown words, use its POS tag

    token2id = {} # dict for convert token to index in total embedding
    id2token = {} # dict for convert index in total embedding to token
    for label in label_dict:
        token2id[label_prefix + label] = len(token2id) # existed label in train data
    token2id[label_prefix + global_null] = len(token2id) # for label is null
    token2id[label_prefix + global_root] = len(token2id) # for root's label
    for pos in pos_dict:
        token2id[pos_prefix + pos] = len(token2id) # existed label in train data
    token2id[pos_prefix + global_null] = len(token2id) # for pos tag is null
    token2id[pos_prefix + global_root] = len(token2id) # for root's pos tag
    token2id[pos_prefix + global_unknown] = len(token2id) # for unknown pos tag
    for dependencies in train_data:
        for dependency in dependencies:
            token2id[dependency[1]] = len(token2id) # set word token's index in total embeddings
    token2id[global_null] = len(token2id) # null word token's index (e.g. some status a word doesn't have left child, etc.)
    token2id[global_root] = len(token2id) # root word's index
    token2id[global_unknown] = len(token2id) # unknown word's index
    for token in token2id:
        id2token[token2id[token]] = token # inverse dict

    len_tokens = len(token2id) # number of total embeddings
    embeddings_matrix = np.asarray(np.random.normal(0, 0.9, (len_tokens, 50)), dtype='float32') # generate embeddings
    for token in token2id:
        i = token2id[token]
        if token in word2vector:
            embeddings_matrix[i] = word2vector[token] # substitute known words to pre-trained word-embeddings
        elif token.lower() in word2vector:
            embeddings_matrix[i] = word2vector[token.lower()] # substitute known words to pre-trained word-embeddings
    return train_data, token2id, id2token, embeddings_matrix
