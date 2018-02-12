from src.data_pro.data_pro import *
import tensorflow as tf
import numpy as np
from src.lsh.lshash import LSHash
from src.model.hiept import HIEPT

data = DataPro(delimiter='\t', ne_type=ne_type)


def hiept():
    # hashing user social network embedding vectors
    encoder_data = data.inputs
    network_lsh = LSHash(network_hash_size, np.shape(encoder_data)[1])
    for v in encoder_data:
        network_lsh.index(v)
        
    network_hamming_code = network_lsh.hamming_code()
    show('network_hamming shape:' + str(np.shape(network_hamming_code)))
    
    # hashing user tweets embedding vectors
    tweets_embedding = data.tweets_embedding
    user_lsh = LSHash(tweets_embedding_hash_size, len(tweets_embedding[0]))
    for embedding in tweets_embedding:
        user_lsh.index(embedding)
    user_hamming_code = user_lsh.hamming_code()
    show('user_hamming shape:' + str(np.shape(user_hamming_code)))
    assert len(network_hamming_code) == len(user_hamming_code)
    
    # concatenate hamming code for tweets and network
    hamming_code = np.hstack((network_hamming_code, user_hamming_code))
    assert len(hamming_code[0]) == network_hash_size + tweets_embedding_hash_size
    
    train_inputs, test_inputs, train_labels, test_labels = split_data(hamming_code, data.labels, rate=0.3, seed=1)
    with tf.Session() as sess:
        with tf.variable_scope('models'):
            hiept = HIEPT(sess, train_inputs, train_labels, test_inputs, test_labels, len(data.labels[0]),
                          lr=0.001, run_time=20000, batch_size=64, drop_out_rate=0.7,
                          learning_rate_decay_factor=0.98)
            sess.run(tf.global_variables_initializer())
            hiept.tran_net()
            predict_results = hiept.predict(test_inputs)
            precession, recall, f1, accuracy = hiept.get_score(predict_results, test_labels)
            show('test results:')
            print 'precession:', precession, 'recall:', recall, 'f1:', f1, 'accuracy:', accuracy
            hiept.show_all_image()


if __name__ == '__main__':
    hiept()
