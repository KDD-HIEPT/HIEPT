import os

data_set = 'douban' # available datasets:douban, weiobo, twitter, zhihu
min_topic_length = 10
max_topic_length = 10000
data_path = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")

original_topic_path = os.path.join(data_path, 'data', data_set, 'topics', "topics.txt")
# graph embedding path
node2vec_path = os.path.join(data_path, 'data', data_set, 'graph_embedding', "node2vec_vec_all.txt")
line_path = os.path.join(data_path, 'data', data_set, 'graph_embedding', "line_vec_all.txt")
tadw_path = os.path.join(data_path, 'data', data_set, 'graph_embedding', "tadw_vec_all.txt")
deepwalk_path = os.path.join(data_path, 'data', data_set, 'graph_embedding', "deepwalk_vec_all.txt")
stuct2vec_path = os.path.join(data_path, 'data', data_set, 'graph_embedding', "stuct2vec_vec_all.txt")
cane_vel_all_path = os.path.join(data_path, 'data', data_set, 'graph_embedding', "cane_vel_all_.txt")

# tweets embedding
tweets_embedding_path = os.path.join(data_path, 'data', data_set, 'tweets_embedding', "tweets_embedding.txt")
embedding_size = 128

# model params
ne_type = 'node2vec'  # available parameters:node2vec, line, tadw, deepwalk, stuct2vec, cane
tweets_embedding_hash_size = 240
row_dim = 28
col_dim = 28
network_hash_size = row_dim * col_dim
final_row_dim = 32
final_col_dim = 32  # note: final_row_dim * final_col_dim = tweets_embedding_hash_size + network_hash_size
lsh_seed = 1
