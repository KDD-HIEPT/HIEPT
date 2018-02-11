import tensorflow as tf
import matplotlib.pyplot as plt
from src.config import *
from src.untils.utils import *


class BaseObject(object):
    def get_score(self, predict, target):
        precession = []
        recall = []
        accuracy = []
        f1 = []
        total_length = len(predict)
        assert total_length == len(target)
        for i in range(total_length):
            predict_number = np.sum(predict[i])
            target_number = np.sum(target[i])
            right_number = self._get_right_number(predict[i], target[i])
            now_precession = 0. if predict_number == 0 else float(right_number * 1. / predict_number)
            now_recall = float(right_number * 1. / target_number)
            precession.append(now_precession)
            recall.append(now_recall)
            f1.append(0 if now_precession == now_recall == 0 else float(2 * now_precession * now_recall/(now_precession + now_recall)))
            accuracy.append(float(right_number*1./np.sum(np.clip((np.array(target[i]) + np.array(predict[i])), a_min=0, a_max=1))))
        return np.mean(precession), np.mean(recall), np.mean(f1), np.mean(accuracy)

    def _get_right_number(self, predict, target):
        right_count = 0
        for i in range(len(predict)):
            if int(predict[i]) + int(target[i]) == 2:
                right_count += 1
        return right_count


class BaseNNObject(BaseObject):
    def _get_batch(self):
        result_x = []
        result_y = []
        if self._current_index + self._batch_size <= self._inputs_num:
            result_x = self._inputs_data[self._current_index:self._current_index + self._batch_size]
            result_y = self._outputs_data[self._current_index:self._current_index + self._batch_size]
            self._current_index += self._batch_size
        else:
            result_x = list(self._inputs_data[self._current_index:])
            result_y = list(self._outputs_data[self._current_index:])
            length = self._batch_size - len(result_x)
            for i in range(length):
                result_x.append(self._inputs_data[i])
                result_y.append(self._outputs_data[i])
            self._current_index = length
        assert len(result_x) == len(result_y) == self._batch_size
        return result_x, result_y

    def _format_labels(self, original_labels):
        final_results = []
        for label in original_labels:
            first_dim = []
            second_dim = []
            for v in label:
                if int(v) == 0:
                    first_dim.append(0)
                    second_dim.append(1)
                elif int(v) == 1:
                    first_dim.append(1)
                    second_dim.append(0)
                else:
                    raise ValueError(str(v) + ' is not 1 or 0')
            final_results.append(np.hstack((first_dim, second_dim)))
        assert len(final_results) == len(original_labels)
        assert len(final_results[0]) == len(original_labels[0]) * 2
        assert np.sum(final_results[-1][len(original_labels[-1]):] + original_labels[-1]) == len(original_labels[-1])
        return final_results

    def _reformat_labels(self, format_labels):
        final_results = []
        for label in format_labels:
            temp_list = []
            first_dim, second_dim = np.hsplit(np.array(label), 2)
            for i in range(len(first_dim)):
                if first_dim[i] > second_dim[i]:
                    temp_list.append(1)
                else:
                    temp_list.append(0)
            final_results.append(temp_list)
        assert len(final_results) == len(format_labels)
        assert len(final_results[-1]) * 2 == len(format_labels[-1])
        return final_results

    def tran_net(self):
        total_loss = 0
        loss_history = [0., 0., 0.]
        for i in range(self._run_time):
            x_batch, y_batch = self._get_batch()
            lr_, _loss, _, _output = self._sess.run([self.learning_rate, self.loss, self.train_op, self._output],
                                                    {self._x: x_batch, self._y: y_batch, self._use_drop_out: True})
            total_loss += _loss
            if i == 0:
                self._test_precision_history.append(0.)
                self._test_recall_history.append(0.)
                self._test_f1_history.append(0.)
                self._test_acc_history.append(0.)
                self._record_index.append(i)
            if i != 0 and i % self._output_interval == 0:
                current_loss = total_loss / self._output_interval
                print 'step:', i, 'lr:', lr_, 'loss:', current_loss
                total_loss = 0
                if i > 1000 and current_loss > max(loss_history):
                    self._sess.run(self.learning_rate_decay_op)
                loss_history[i % 3] = current_loss
                # record
                train_p_, train_r_, train_f1_, train_acc_ = self.get_score(self.predict(self._inputs_data),
                                                               self._original_outputs_data)
                self._train_precision_history.append(train_p_)
                self._train_recall_history.append(train_r_)
                self._train_f1_history.append(train_f1_)
                self._train_acc_history.append(train_acc_)
                self._record_index.append(i)
                self._loss_history.append(current_loss)
                test_p_, test_r_, test_f1_, test_acc_ = self.get_score(self.predict(self._test_inputs_data),
                                                            self._test_labels_data)
                self._test_precision_history.append(test_p_)
                self._test_recall_history.append(test_r_)
                self._test_f1_history.append(test_f1_)
                self._test_acc_history.append(test_acc_)

    def predict(self, inputs):
        return self._reformat_labels(self._sess.run(self._output, {self._x: inputs, self._use_drop_out: False}))

    def show_all_image(self):
        line_width = 3.0
        font_size = 26
        line1, = plt.plot(self._record_index, self._test_precision_history, color='fuchsia', label="macro-P",
                          linestyle='-', linewidth=line_width, marker='+')
        line2, = plt.plot(self._record_index, self._test_f1_history, color='blue', label="macro-F1",
                          linestyle='-', linewidth=line_width, marker='x')
        line3, = plt.plot(self._record_index, self._test_acc_history, color='green', label="Accuracy",
                          linestyle='-', linewidth=line_width, marker='|')
        plt.legend(handles=[line1, line2, line3], loc=4, fontsize=23)
        plt.xlabel('Iteration', fontsize=font_size)
        plt.ylabel('Value', fontsize=font_size)
        plt.ylim(0.0, 0.7)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # show image
        plt.show()


class HIEPT(BaseNNObject):
    def __init__(self, sess, inputs_data, outputs_data, test_inputs, test_labels, output_dim, lr=0.0001,
                 batch_size=128, run_time=100000, learning_rate_decay_factor=0.98, output_interval=200,
                 drop_out_rate=0.9):
        self._sess = sess
        self._drop_out_rate = drop_out_rate
        self._inputs_data = inputs_data
        self._output_dim = output_dim * 2
        self._original_outputs_data = outputs_data
        self._outputs_data = self._format_labels(outputs_data)
        self._test_inputs_data = test_inputs
        self._test_labels_data = test_labels
        self._x = tf.placeholder(tf.float32, [None, len(inputs_data[0])])
        self._y = tf.placeholder(tf.int32, [None, self._output_dim])
        self._use_drop_out = tf.placeholder(tf.bool, None)
        self._inputs_num = len(self._inputs_data)
        self.learning_rate = tf.Variable(
            float(lr), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self._current_index = 0
        self._batch_size = batch_size
        self._run_time = run_time
        self._output_interval = output_interval
        # record
        self._record_index = []
        self._loss_history = []
        self._train_precision_history = []
        self._train_recall_history = []
        self._train_f1_history = []
        self._train_acc_history = []
        self._test_precision_history = []
        self._test_recall_history = []
        self._test_f1_history = []
        self._test_acc_history = []

        self._build_net()

    def _build_net(self):
        self.l0 = tf.layers.dense(self._x, final_row_dim * final_row_dim, tf.nn.tanh)
        inputs = tf.reshape(self.l0, [-1, final_row_dim, final_row_dim, 1])  # (batch, height, width, channel)

        # CNN
        conv1 = tf.layers.conv2d(  # shape (32, 32, 1)
            inputs=inputs,
            filters=16,
            kernel_size=5,
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )  # -> (32, 32, 16)
        pool1 = tf.layers.average_pooling2d(
            conv1,
            pool_size=2,
            strides=2,
        )  # -> (16, 16, 16)
        conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)  # -> (16, 16, 32)
        pool2 = tf.layers.average_pooling2d(conv2, 2, 2)  # -> (8, 8, 32)
        flat = tf.reshape(pool2, [-1, 8 * 8 * 32])  # -> (8*8*32, )
        temp_output = tf.layers.dense(flat, self._output_dim)  # output layer
        self._output = tf.layers.dropout(temp_output, rate=self._drop_out_rate, training=self._use_drop_out)

        self.loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self._y, logits=self._output)  # compute cost
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)