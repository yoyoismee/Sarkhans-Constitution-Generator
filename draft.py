
import tensorflow as tf
import numpy as np
import argparse
import pickle
parser = argparse.ArgumentParser()

parser.add_argument('len',
                    type=int,
                    help='Constitution length (in word)')
parser.add_argument('--seed',
                    type=str,
                    help='seed word',
                    default = "ประชาชน")

args = parser.parse_args()

_, vocab_to_int, int_to_vocab, token_dict = pickle.load(open('data.pj', mode='rb'))
seq_length, load_dir = pickle.load(open('params.pj', mode='rb'))


def get_tensors(loaded_graph):

    InputTensor = loaded_graph.get_tensor_by_name("input:0")
    InitialStateTensor = loaded_graph.get_tensor_by_name("initial_state:0")
    FinalStateTensor = loaded_graph.get_tensor_by_name("final_state:0")
    ProbsTensor = loaded_graph.get_tensor_by_name("probs:0")
    # TODO: Implement Function
    return (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)

def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    # TODO: Implement Function
    return int_to_vocab[np.random.choice(range(len(probabilities)),size=1,p=probabilities)[0]]

gen_length = args.len
# homer_simpson, moe_szyslak, or Barney_Gumble
prime_word = args.seed

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word]
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})
        
        pred_word = pick_word(probabilities[0][dyn_seq_length-1], int_to_vocab)

        gen_sentences.append(pred_word)
    
    # Remove tokens
    tv_script = ''.join(gen_sentences)
    for key, token in token_dict.items():
        ending = '' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace('' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')
    tv_script = tv_script.replace('[space]', ' ')

    print(tv_script)
