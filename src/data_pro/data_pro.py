import numpy as np

from src.config import *
from src.untils.utils import *


class DataPro(object):
    def __init__(self, is_save=False, delimiter='\t', ne_type=''):
        self._is_save = is_save
        self._new_topics = []
        self._delimiter = delimiter
        self._ne_type = ne_type
        self._data_pre_process()

    def _data_pre_process(self):
        show('data process begin')
        # get the mapping dic
        f = open(original_topic_path, 'r')
        temp_set = set()
        for line in f.readlines():
            line_set = set(line.replace('\n', '').split(self._delimiter)[:-1])
            if max_topic_length >= len(line_set) >= min_topic_length:
                temp_set = temp_set.union(line_set)
                self._new_topics.append(map(int, list(line_set)))
        f.close()
        self._mapping_dic = sorted(map(int, list(temp_set)))
        # get labels
        self._labels = np.zeros([len(self._mapping_dic), len(self._new_topics)], dtype=int)
        show('total nodes:' + str(len(self._mapping_dic)))
        for index, topic in enumerate(self._new_topics):
            for person in topic:
                self._labels[binary_search(self._mapping_dic, person)][index] = 1
        # get inputs
        self._inputs = []
        self._tweets_embedding = []
        ne_dic = {}
        if 'node2vec' == self._ne_type:
            f_o = open(node2vec_path, 'r')
        elif 'line' == self._ne_type:
            f_o = open(line_path, 'r')
        elif 'tadw' == self._ne_type:
            f_o = open(tadw_path, 'r')
        elif 'deepwalk' == self._ne_type:
            f_o = open(deepwalk_path, 'r')
        elif 'stuct2vec' == self._ne_type:
            f_o = open(stuct2vec_path, 'r')
        elif 'cane' == self._ne_type:
            f_o = open(cane_vel_all_path, 'r')
        else:
            raise ValueError('ne_type must have value')
        for line in f_o.readlines():
            if '#' in line:
                continue
            if 'stuct2vec' == self._ne_type:
                lines = line.replace('\n', '').split(' ')[:-1]
            else:
                lines = line.replace('\n', '').split(' ')
            # np.random.standard_normal(200) if only for cane, in case that some user have not friends
            ne_dic[str(lines[0])] = np.random.standard_normal(200) if len(lines[1:]) == 1 else map(float, list(
                lines[1:]))
        f_o.close()
        # get user embedding
        tweets_embedding_dic = load_list_dic(tweets_embedding_path, dtype='float', tag='user embedding')
        for node in self._mapping_dic:
            self._inputs.append(ne_dic[str(node)])
            # self._inputs.append(open_ne_dic[str(node)] if key_in_dic(str(node), open_ne_dic) else
            #                     [float(0) for i in range(256)])
            self._tweets_embedding.append(tweets_embedding_dic[str(node)])
        show('total communities:' + str(len(self._labels[0])))
        show('data process finished')

    @property
    def labels(self):
        return self._labels

    @property
    def inputs(self):
        return self._inputs

    @property
    def tweets_embedding(self):
        return self._tweets_embedding
