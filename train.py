
import numpy as np
import argparse
import pickle
parser = argparse.ArgumentParser()


parser.add_argument('file',
                    type=str,
                    help='input file')


data_dir = parser.parse_args().file
with open(data_dir, "r") as f:
    text = f.read()


scenes = text.split('\n\n')
sentence_count_scene = [scene.count('\n') for scene in scenes]

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
word_count_sentence = [len(sentence.split()) for sentence in sentences]


def create_lookup_tables(text):

    w2i = {}
    i2w = {}
    for i , word in enumerate(set(text)):
        w2i[word] = i
        i2w[i] = word
    return w2i, i2w



###

token_dict = {".":"|dot|",",":"|comma|","\"":"|qoute|",";":"|semicolon|","!":"|exclamation|","?":"|question|","(":"|patenthesesL|",")":"|patenthesesR|","--":"|dash|","\n":"|return|",}
for key, token in token_dict.items():
    text = text.replace(key, ' {} '.format(token))

text = text.lower()
text = text.split()

vocab_to_int, int_to_vocab = create_lookup_tables(text)
int_text = [vocab_to_int[word] for word in text]
pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('data.pj', 'wb'))

###

int_text, vocab_to_int, int_to_vocab, token_dict = pickle.load(open('data.pj', mode='rb'))

import warnings
import tensorflow as tf


def get_inputs():
    Input = tf.placeholder(dtype=tf.int32,shape=[None,None],name="input")
    Targets = tf.placeholder(dtype=tf.int32,shape=[None,None],name="targets")
    LearningRate = tf.placeholder(dtype=tf.float32,name="learningrate")
    return Input, Targets, LearningRate



def get_init_cell(batch_size, rnn_size):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    cell = tf.contrib.rnn.MultiRNNCell([lstm])
    initial_state = cell.zero_state(batch_size, tf.float32)
    initial_state = tf.identity(input=initial_state,name='initial_state')
    return cell, initial_state


def get_embed(input_data, vocab_size, embed_dim):
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    embeded = tf.nn.embedding_lookup(embedding, input_data)
    return embeded


def build_rnn(cell, inputs):
    outputs, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs,dtype=tf.float32)
    final_state = tf.identity(input=final_state,name="final_state")
    # TODO: Implement Function
    return outputs, final_state

def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):

    embeded = get_embed(embed_dim = embed_dim, input_data = input_data,vocab_size = vocab_size)
    outputs, final_state = build_rnn(cell=cell, inputs= embeded)
    logits = tf.contrib.layers.fully_connected(inputs=outputs,num_outputs= vocab_size, activation_fn=None)
    print(final_state.shape)
    return logits, final_state

def get_batches(int_text, batch_size, seq_length):
    n_batch = len(int_text)//(batch_size*seq_length)
    out = np.ndarray((n_batch, 2, batch_size, seq_length))
    maxtar = n_batch * batch_size * seq_length
    for i in range(n_batch):
        for j in range(batch_size):
            out[i][0][j]=(int_text[(j*(n_batch*seq_length)+i*(seq_length)):(j*(n_batch*seq_length)+(i+1)*(seq_length))])
            out[i][1][j]=(int_text[(j*(n_batch*seq_length)+i*(seq_length)+1):(j*(n_batch*seq_length)+(i+1)*(seq_length)+1)])
            if (j*(n_batch*seq_length)+(i+1)*(seq_length)+1) > maxtar:
                out[i][1][j][-1] = int_text[0]
    return out

# Number of Epochs
num_epochs = 99
# Batch Size
batch_size = 256
# RNN Size
rnn_size = 256
# Embedding Dimension Size
embed_dim = 100
# Sequence Length
seq_length = 15
# Learning Rate
learning_rate = 0.01
# Show stats for every n number of batches
show_every_n_batches = 100

save_dir = './tmp'

from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)


batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')
pickle.dump((seq_length, save_dir), open('params.pj', 'wb'))