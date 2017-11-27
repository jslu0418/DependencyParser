import sys, traceback
import tensorflow as tf
import numpy as np
from time import time
from prepare_data import prepare_data
from dependent_parser import extract_features_from_train_data
from utils import format_string, print_title

pos_prefix = '<pos> :'
label_prefix = '<label> :'
global_root = '<root>'
global_unknown = '<unknown>'
unknown_prefix = '<unknown> :'
global_null = '<null>'

def tfcube(x):
    return tf.matmul(tf.matmul(x, tf.matrix_transpose(x)),x)

dev_mode = 1
optimizer = {'AdamOptimizer': tf.train.AdamOptimizer,'GradientDescentOptimizer': tf.train.GradientDescentOptimizer}
activation = {'cube': tfcube, 'sigmoid': tf.nn.sigmoid, 'tanh': tf.nn.tanh, 'relu': tf.nn.relu}
# Parameters
batch_size = int(sys.argv[1]) # batch size
valid_batch_size = int(sys.argv[2]) # valid batch size
hidden_size = int(sys.argv[3])
dropout = float(sys.argv[4])
optimizerIndex = int(sys.argv[5])
activationIndex = int(sys.argv[6])
trainsteps = int(sys.argv[7])
extract_feature_time = 0
training_time = 0

useOptimizer = optimizer[list(optimizer.keys())[optimizerIndex]]
useActivation = activation[list(activation.keys())[activationIndex]]
print(useOptimizer)
print(useActivation)
embedding_size = 50 # embedding width
features_size = 48 # feature size, 48 features

before = time()
train_data, token2id, id2token, embeddings_matrix = prepare_data('train') # get train data
print('get train data in {}s'.format(time()-before))
before = None

if dev_mode == 1:
    # if in dev_mode
    dev_data, dev_token2id, dev_id2token, _ = prepare_data('dev') # get dev data
    dev_sentence_index = 0
    len_dev_dependecies = len(dev_data)


# For classifier in neural network
class2label = {} # dict for convert class to label
label2class = {} # dict for convert label to class
for i in range(len(id2token)):
    if id2token[i] == label_prefix + global_root:
        break
    elif id2token[i] == label_prefix + global_null:
        class2label['shift' + ':' + global_null] = len(class2label) # shift label
    else:
        class2label['left-arc' + ':' + id2token[i][len(label_prefix):]] = len(class2label) # left arc label
        class2label['right-arc' + ':' + id2token[i][len(label_prefix):]] = len(class2label) # right arc label

for k in class2label:
    label2class[class2label[k]] = k # inverse conversion


labels_classes_size = len(label2class) # total kind of classes
sentence_index = 0 # record sentence index
len_total_dependecies = len(train_data) # record how many sentence in train_data

def one_hot(i):
    '''generate one hot vector'''
    a = np.zeros(labels_classes_size, np.float32)
    a[i] = 1.0
    return a


def generate_dev_batch(valid_batch_size):
    '''produce dependencies batch for deving'''
    global dev_sentence_index # dev_sentence_index
    batch_data = [] # initialize batch_data list
    batch_labels = [] # initialize batch_labels list
    status_lists = [] # this batch statuslists
    while len(batch_data) < valid_batch_size:
        # batch_data's length less than goal
        if dev_sentence_index >= len_dev_dependecies:
            # surpass the total length of dev_data
            dev_sentence_index = dev_sentence_index - len_dev_dependecies # substract total length of dev_data
        data_in_sentence, one_status_lists = extract_features_from_train_data(dev_data[dev_sentence_index], token2id) # extract every status' features in one sentence
        dev_sentence_index += 1 # increase dev sentence index
        batch_data += [e['features'] for e in data_in_sentence] # add several status' features
        batch_labels += [one_hot(class2label[e['op'] + ':' + e['label'][len(label_prefix):]]) for e in data_in_sentence] # add several status' labels
        status_lists = status_lists + one_status_lists
    if len(batch_data) > valid_batch_size:
        # if over 2048, delete redundant
        batch_data = batch_data[:valid_batch_size]
        batch_labels = batch_labels[:valid_batch_size]
        dev_sentence_index -= 1
    return batch_data, batch_labels, status_lists

def generate_batch(batch_size):
    '''produce dependencies batch for training'''
    global extract_feature_time
    before = time()
    global sentence_index # sentence_index
    batch_data = [] # initialize batch_data list
    batch_labels = [] # initialize batch_labels list
    while len(batch_data) < batch_size:
        # batch_data's length less than goal
        if sentence_index >= len_total_dependecies:
            # surpass the total length of train_data
            sentence_index = sentence_index - len_total_dependecies # substract total length of train_data
        data_in_sentence, _= extract_features_from_train_data(train_data[sentence_index], token2id) # extract every status' features in one sentence
        sentence_index += 1 # increase dev sentence index
        batch_data += [e['features'] for e in data_in_sentence] # add several status' features
        batch_labels += [one_hot(class2label[e['op'] + ':' + e['label'][len(label_prefix):]]) for e in data_in_sentence] # add several status' labels
    if len(batch_data) > batch_size:
        # if over 2048, delete redundant
        batch_data = batch_data[:batch_size]
        batch_labels = batch_labels[:batch_size]
        sentence_index -= 1
    extract_feature_time = time() - before
    return batch_data, batch_labels


# tensorflow related started here.
graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[None, features_size])
    train_labels = tf.placeholder(tf.float32, shape=[None, labels_classes_size])

    # lookup embeddings
    embeds = tf.nn.embedding_lookup(embeddings_matrix , train_inputs)
    # split to three kind of features
    w_embeds, p_embeds, l_embeds = tf.split(embeds, [18, 18, 12], 1)
    # reshape train data embeddings to batch size rows
    w_embeds = tf.reshape(w_embeds, [batch_size, 18 * 50])
    p_embeds = tf.reshape(p_embeds, [batch_size, 18 * 50])
    l_embeds = tf.reshape(l_embeds, [batch_size, 12 * 50])
    # weights for different kind of features
    if activationIndex == 1:
        w_weights = tf.Variable(tf.random_uniform([18 * 50, hidden_size]))
        p_weights = tf.Variable(tf.random_uniform([18 * 50, hidden_size]))
        l_weights = tf.Variable(tf.random_uniform([12 * 50, hidden_size]))
        biases = tf.Variable(tf.zeros([hidden_size])) # biases
        # weights for second layer
        weights2 = tf.Variable(tf.zeros([hidden_size, labels_classes_size]))
        biases2 = tf.Variable(tf.zeros([labels_classes_size])) # biases
    else:
        w_weights = tf.Variable(tf.random_uniform([18 * 50, hidden_size],minval=0, maxval=1))
        p_weights = tf.Variable(tf.random_uniform([18 * 50, hidden_size],minval=0, maxval=1))
        l_weights = tf.Variable(tf.random_uniform([12 * 50, hidden_size],minval=0, maxval=1))
        biases = tf.Variable(tf.random_uniform([hidden_size], minval=0, maxval=1)) # biases
        weights2 = tf.Variable(tf.random_uniform([hidden_size, labels_classes_size], minval=0, maxval=1))
        biases2 = tf.Variable(tf.random_uniform([labels_classes_size], minval=0, maxval=1)) # biases
    # activation function sigmoid
    h_1 = useActivation(tf.matmul(w_embeds, w_weights) + tf.matmul(p_embeds, p_weights) + tf.matmul(l_embeds, l_weights) + biases)
    # drop, deal with overfitting
    h_1_drop = tf.nn.dropout(h_1, dropout)
    y = tf.matmul(h_1_drop, weights2) + biases2

    # cross_entropy loss function
    cross_entropy = tf.reduce_mean(
#        tf.nn.softmax_cross_entropy_with_logits(labels=train_labels, logits=y)
        tf.losses.softmax_cross_entropy(onehot_labels=train_labels, logits=y)
    )

    # optimizer
    train_step = useOptimizer(0.01).minimize(cross_entropy)

    # valid part
    valid_embeds = tf.nn.embedding_lookup(embeddings_matrix , train_inputs)
    # split to three kind of features
    valid_w_embeds, valid_p_embeds, valid_l_embeds = tf.split(embeds, [18, 18, 12], 1)
    # reshape valid data embeddings to valid batch size rows
    valid_w_embeds = tf.reshape(valid_w_embeds, [valid_batch_size, 18 * 50])
    valid_p_embeds = tf.reshape(valid_p_embeds, [valid_batch_size, 18 * 50])
    valid_l_embeds = tf.reshape(valid_l_embeds, [valid_batch_size, 12 * 50])
    # active function sigmoid
    valid_h_1 = useActivation(tf.matmul(valid_w_embeds, w_weights) + tf.matmul(valid_p_embeds, p_weights) + tf.matmul(valid_l_embeds, l_weights) + biases)
    # drop
    valid_h_1_drop = tf.nn.dropout(valid_h_1, dropout)
    valid_y = tf.matmul(valid_h_1_drop, weights2)

    # prediction label
    correct_prediction = tf.equal(tf.argmax(train_labels,1), tf.argmax(valid_y ,1))

    # accuracy evaluation
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()



steps = trainsteps
with tf.Session(graph=graph) as sess:
    init.run()
    for step in range(steps):
        batch_data, batch_label = generate_batch(batch_size) # generate train batch
        # every 500 steps print time and loss Value
        if step % 500 == 1:
            if before is None:
                before = time()
            else:
                print('finish 500 steps in {}s.'.format(time()-before))
                before = time()
                # print this step's corss_entropy
                print(sess.run(cross_entropy, feed_dict={train_inputs:batch_data, train_labels: batch_label}))
                print('valid step start:')
                steps = 5 # number of validation steps
                # validation steps
                for step in range(steps):
                    dev_batch_data, dev_batch_label, slists = generate_dev_batch(valid_batch_size)
                    print(accuracy.eval(feed_dict={train_inputs: dev_batch_data, train_labels: dev_batch_label}))
                    prediction_labels = sess.run(tf.argmax(valid_y, 1), feed_dict={train_inputs:dev_batch_data, train_labels: dev_batch_label})

                    cur_sentence = {}
                    for i in range(valid_batch_size):
                        if len(cur_sentence) != 0 and slists[i][0] == None:
                            # current status list is a new sentence's initial status (stack is empty)
                            order = list(cur_sentence.keys())
                            # sort by key
                            order.sort()
                            print_title()
                            for i in order:
                                # print this result of dependency parsing for current sentence
                                print(cur_sentence[i])
                            cur_sentence = {}
                        pre_label = label2class[prediction_labels[i]] # get prediction label's name
                        if pre_label != 'shift:' + global_null:
                            # if not a shift operation, means this operation denote a dependency
                            arclabel = pre_label.split(':')
                            if arclabel[0] == 'left-arc':
                            # stack[-2] depends on stack[-1]
                                cld = slists[i][1] # stack[-2]
                                prt = slists[i][0] # stack[-1]
                                if cld is not None and prt is not None:
                                    # append this to cur_sentence
                                    cur_sentence[int(cld[0])] = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(cld[0],format_string(cld[1]),cld[4],cld[6],format_string(cld[7]),prt[0],format_string(arclabel[1]),'Matched' if cld[6]==prt[0] else 'Not', 'Matched' if cld[7] == arclabel[1] else 'Not')
                            else:
                            # stack[-1] depends on stack[-2] (root dependency included at here)
                                cld = slists[i][0] # stack[-1]
                                prt = slists[i][1] # stack[-2]
                                if prt is None:
                                    prt = ['0','0'] # parent is Root default
                                # append this to cur_sentence
                                if cld is not None:
                                    cur_sentence[int(cld[0])] = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(cld[0],format_string(cld[1]),cld[4],cld[6],format_string(cld[7]),prt[0],format_string(arclabel[1]),'Matched' if cld[6]==prt[0] else 'UnMatch', 'Not' if cld[7] == arclabel[1] else 'Not')

                print('valid step stop')

        sess.run(train_step, feed_dict={train_inputs: batch_data, train_labels: batch_label})
    print('extract feature total used {}s, feature for every 500 steps use {}s.'.format(extract_feature_time, extract_feature_time*500/trainsteps))
