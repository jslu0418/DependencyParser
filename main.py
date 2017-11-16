import sys, traceback
import tensorflow as tf
import numpy as np
from time import time
from prepare_data import prepare_data
from dependent_parser import extract_features_from_train_data

pos_prefix = '<pos> :'
label_prefix = '<label> :'
global_root = '<root>'
global_unknown = '<unknown>'
unknown_prefix = '<unknown> :'
global_null = '<null>'

dev_mode = 1
optimizer = {'AdamOptimizer': tf.train.AdamOptimizer,'GradientDescentOptimizer': tf.train.GradientDescentOptimizer}
# Parameters
batch_size = 512 # batch size
valid_batch_size = 128 # valid batch size
embedding_size = 50 # embedding width
features_size = 48 # feature size, 48 features

before = time()
train_data, token2id, id2token, embeddings_matrix = prepare_data('train') # get train data
print('get train data in {}'.format(time()-before))
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
    while len(batch_data) < valid_batch_size:
        # batch_data's length less than goal
        if dev_sentence_index >= len_dev_dependecies:
            # surpass the total length of dev_data
            dev_sentence_index = dev_sentence_index - len_dev_dependecies # substract total length of dev_data
        data_in_sentence = extract_features_from_train_data(dev_data[dev_sentence_index], token2id) # extract every status' features in one sentence
        dev_sentence_index += 1 # increase dev sentence index
        batch_data += [e['features'] for e in data_in_sentence] # add several status' features
        batch_labels += [one_hot(class2label[e['op'] + ':' + e['label'][len(label_prefix):]]) for e in data_in_sentence] # add several status' labels
    if len(batch_data) > valid_batch_size:
        # if over 2048, delete redundant
        batch_data = batch_data[:valid_batch_size]
        batch_labels = batch_labels[:valid_batch_size]
        dev_sentence_index -= 1
    return batch_data, batch_labels

def generate_batch(batch_size):
    '''produce dependencies batch for training'''
    global sentence_index # sentence_index
    batch_data = [] # initialize batch_data list
    batch_labels = [] # initialize batch_labels list
    while len(batch_data) < batch_size:
        # batch_data's length less than goal
        if sentence_index >= len_total_dependecies:
            # surpass the total length of train_data
            sentence_index = sentence_index - len_total_dependecies # substract total length of train_data
        data_in_sentence = extract_features_from_train_data(train_data[sentence_index], token2id) # extract every status' features in one sentence
        sentence_index += 1 # increase dev sentence index
        batch_data += [e['features'] for e in data_in_sentence] # add several status' features
        batch_labels += [one_hot(class2label[e['op'] + ':' + e['label'][len(label_prefix):]]) for e in data_in_sentence] # add several status' labels
    if len(batch_data) > batch_size:
        # if over 2048, delete redundant
        batch_data = batch_data[:batch_size]
        batch_labels = batch_labels[:batch_size]
        sentence_index -= 1
    return batch_data, batch_labels


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
    w_weights = tf.Variable(tf.zeros([18 * 50, 111]))
    p_weights = tf.Variable(tf.zeros([18 * 50, 111]))
    l_weights = tf.Variable(tf.zeros([12 * 50, 111]))
    biases = tf.Variable(tf.zeros([111])) # biases
    # weights for second layer
    weights2 = tf.Variable(tf.zeros([111, labels_classes_size]))
    # active function sigmoid
    h_1 = tf.nn.sigmoid(tf.matmul(w_embeds, w_weights) + tf.matmul(p_embeds, p_weights) + tf.matmul(l_embeds, l_weights) + biases)
    # drop
    h_1_drop = tf.nn.dropout(h_1, 0.5)
    y = tf.matmul(h_1_drop, weights2)

    # cross_entropy loss function
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=train_labels, logits=y)
    )

    # optimizer
    train_step = optimizer['AdamOptimizer'](0.001).minimize(cross_entropy)

    # valid part
    valid_embeds = tf.nn.embedding_lookup(embeddings_matrix , train_inputs)
    # split to three kind of features
    valid_w_embeds, valid_p_embeds, valid_l_embeds = tf.split(embeds, [18, 18, 12], 1)
    # reshape valid data embeddings to valid batch size rows
    valid_w_embeds = tf.reshape(valid_w_embeds, [valid_batch_size, 18 * 50])
    valid_p_embeds = tf.reshape(valid_p_embeds, [valid_batch_size, 18 * 50])
    valid_l_embeds = tf.reshape(valid_l_embeds, [valid_batch_size, 12 * 50])
    # active function sigmoid
    valid_h_1 = tf.nn.sigmoid(tf.matmul(valid_w_embeds, w_weights) + tf.matmul(valid_p_embeds, p_weights) + tf.matmul(valid_l_embeds, l_weights) + biases)
    # drop
    valid_h_1_drop = tf.nn.dropout(valid_h_1, 0.5)
    valid_y = tf.matmul(valid_h_1_drop, weights2)

    correct_prediction = tf.equal(tf.argmax(train_labels,1), tf.argmax(valid_y ,1))

    # accuracy evaluation
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()



steps = 501
with tf.Session(graph=graph) as sess:
    init.run()
    for step in range(steps):
        batch_data, batch_label = generate_batch(batch_size)
        # every 500 steps print time and loss_function value
        if step % 500 == 1:
            if before is None:
                before = time()
            else:
                print('finish 500 steps in {}s.'.format(time()-before))
                before = time()
                print(sess.run(cross_entropy, feed_dict={train_inputs:batch_data, train_labels: batch_label}))
        sess.run(train_step, feed_dict={train_inputs: batch_data, train_labels: batch_label})

    print('valid step:')
    steps = 5
    # validation steps
    for step in range(steps):
        dev_batch_data, dev_batch_label = generate_dev_batch(valid_batch_size)
        print(accuracy.eval(feed_dict={train_inputs: dev_batch_data, train_labels: dev_batch_label}))
