# coding=utf-8
import random
import time
import numpy as np


def show(msg):
    print "[INFO]", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), msg


def save_2d_matrix(matrix, save_path, delimiter='\t', tag=''):
    f = open(save_path, 'w')
    for line in matrix:
        for value in line:
            f.write(str(value))
            f.write(delimiter)
        f.write('\n')
    f.close()
    show(tag + ' matrix saved')


def load_2d_matrix(matrix_path, delimiter='\t', dtype='int', tag=''):
    result = []
    f = open(matrix_path, 'r')
    for line in f.readlines():
        lines = line.split(delimiter)[:-1]
        if dtype == 'int':
            lines = map(int, lines)
        elif dtype == 'float':
            lines = map(float, lines)
        else:
            raise ValueError(str(dtype) + ' not supported')
        result.append(lines)
    show(tag + ' matrix loaded')
    return result


def save_list_dic(dic, save_path, delimiter='\t', tag=''):
    f = open(save_path, 'w')
    for key in dic.keys():
        f.write(str(key))
        for v in list(dic[key]):
            f.write(delimiter + str(v))
        f.write('\n')
    f.close()
    show(tag + ' dic saved')


def save_list_in_one_dic(dic, save_path, delimiter='\t', tag=''):
    f = open(save_path, 'w')
    for key in dic.keys():
        f.write(str(key)+delimiter)
        for v in list(dic[key]):
            f.write(str(v))
        f.write('\n')
    f.close()
    show(tag + ' dic saved')


def load_list_dic(dic_path, delimiter='\t', dtype='int', tag=''):
    f = open(dic_path)
    dic = {}
    for line in f.readlines():
        lines = line.replace('\n', '').split(delimiter)
        if dtype == 'int':
            dic[lines[0]] = map(int, lines[1:])
        elif dtype == 'float':
            dic[lines[0]] = map(float, lines[1:])
        else:
            raise ValueError(str(dtype) + ' not supported')
    f.close()
    return dic


def binary_search(source, value):
    """
    the source should be a sorted list
    binary search,return the index of value if value in source,else return None
    :param source:a sorted list
    :param value:the element to find
    :return:
    """
    low = 0
    high = len(source) - 1
    while low <= high:
        mid = (low + high) / 2
        mid_value = source[mid]
        if value == mid_value:
            return mid
        elif value < mid_value:
            high = mid - 1
        else:
            low = mid + 1
    return None


def split_data(inputs, labels, rate=0.3, seed=1, need_index=False):
    """
    :param need_index: if True, return train and test index
    :param inputs: total inputs,It should be an array
    :param labels: total labels,It should be an array
    :param rate: len(test_data)/len(total_data)
    :param seed: if not zero,it will get same results every time
    :return:train_inputs, test_inputs, train_labels, test_labels
    """
    train_inputs = []
    train_labels = []
    test_inputs = []
    test_labels = []
    total_length = len(inputs)
    assert total_length == len(labels)
    random.seed(seed)
    index = [i for i in range(total_length)]
    test_index = random.sample(index, int(float(total_length) * rate))
    train_index = list(set(index).difference(set(test_index)))
    for tr_index in train_index:
        train_inputs.append(inputs[tr_index])
        train_labels.append(labels[tr_index])
    for ts_index in test_index:
        test_inputs.append(inputs[ts_index])
        test_labels.append(labels[ts_index])
    show('train length:' + str(len(train_inputs)))
    show('test length:' + str(len(test_inputs)))
    if need_index:
        return train_inputs, test_inputs, train_labels, test_labels, train_index, test_index
    else:
        return train_inputs, test_inputs, train_labels, test_labels


def array_2d_to_string(array):
    result = []
    for v in array:
        new_str = ''
        for vv in v:
            new_str += str(vv)
        result.append(new_str)
    return result


def string_to_2d_array(str):
    result = []
    for s in str:
        result.append(map(int, list(s)))
    return result


def quick_sort(array):
    if len(array) < 2:
        return array
    else:
        # recursive case
        pivot = array[0]
        # sub-array of all the elements less than the pivot
        less = [i for i in array[1:] if i <= pivot]
        # sub-array of all the elements greater than the pivot
        greater = [i for i in array[1:] if i > pivot]
        return quick_sort(greater) + [pivot] + quick_sort(less)


def top_n_index(array, n):
    array_list = list(array)
    sorted_array = quick_sort(array)
    return [array_list.index(i) for i in sorted_array[:n]]


def min_n_index(array, n):
    array_list = list(array)
    sorted_array = quick_sort(array)
    return [array_list.index(i) for i in sorted_array[-n:]]


def most_similar_labels(predict_label, total_lables):
    temp_list = []
    for label in total_lables:
        temp_list.append(np.sum(map(abs, predict_label - label)))
    return total_lables[np.argmin(temp_list)]


def save_list(ll, save_path):
    f = open(save_path, 'w')
    for v in ll:
        f.write(str(v) + '\n')
    f.close()


def load_list(save_path):
    result = []
    f = open(save_path, 'r')
    for line in f.readlines():
        result.append(float(line.replace('\n', '')))
    f.close()
    return result


def timestamp_to_datetime(value):
    time_format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(time_format, value)
    return dt


def datetime_to_timestamp(dt):
    time.strptime(dt, '%Y-%m-%d %H:%M:%S')
    s = time.mktime(time.strptime(dt, '%Y-%m-%d %H:%M:%S'))
    return int(s)
