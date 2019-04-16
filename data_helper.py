# -*- coding:utf-8 -*-

from bert_serving.client import BertClient
import os
import csv
import tokenization
import tensorflow as tf
import numpy as np
import json


# Yelp_data_path = 'yelp_new_reviews'
# TRAIN_WORD_EMBEDDING_PATH = 'yelp_new_reviews/train_embedding_word.npy'
# DEV_WORD_EMBEDDING_PATH = 'yelp_new_reviews/dev_embedding_word.npy'
# TEST_WORD_EMBEDDING_PATH = 'yelp_new_reviews/test_embedding_word.npy'
# TRAIN_SENTENCE_EMBEDDING_PATH = 'yelp_new_reviews/train_embedding_sen.npy'
# DEV_SENTENCE_EMBEDDING_PATH = 'yelp_new_reviews/dev_embedding_sen.npy'
# TEST_SENTENCE_EMBEDDING_PATH = 'yelp_new_reviews/test_embedding_sen.npy'
# TRAIN_LABEL_PATH = 'yelp_new_reviews/train_label.npy'
# DEV_LABEL_PATH = 'yelp_new_reviews/dev_label.npy'
# TEST_LABEL_PATH = 'yelp_new_reviews/test_label.npy'


# Amazon_data_path = 'Amazon_mobile_reviews'
# TRAIN_WORD_EMBEDDING_PATH = 'Amazon_mobile_reviews/train_embedding_word.npy'
# DEV_WORD_EMBEDDING_PATH = 'Amazon_mobile_reviews/dev_embedding_word.npy'
# TEST_WORD_EMBEDDING_PATH = 'Amazon_mobile_reviews/test_embedding_word.npy'
# TRAIN_SENTENCE_EMBEDDING_PATH = 'Amazon_mobile_reviews/train_embedding_sen.npy'
# DEV_SENTENCE_EMBEDDING_PATH = 'Amazon_mobile_reviews/dev_embedding_sen.npy'
# TEST_SENTENCE_EMBEDDING_PATH = 'Amazon_mobile_reviews/test_embedding_sen.npy'
# TRAIN_LABEL_PATH = 'Amazon_mobile_reviews/train_label.npy'
# DEV_LABEL_PATH = 'Amazon_mobile_reviews/dev_label.npy'
# TEST_LABEL_PATH = 'Amazon_mobile_reviews/test_label.npy'



# Amazon_food_path = 'amazon_food_reviews'
# TRAIN_WORD_EMBEDDING_PATH = 'amazon_food_reviews/train_embedding_word.npy'
# DEV_WORD_EMBEDDING_PATH = 'amazon_food_reviews/dev_embedding_word.npy'
# TEST_WORD_EMBEDDING_PATH = 'amazon_food_reviews/test_embedding_word.npy'
# TRAIN_SENTENCE_EMBEDDING_PATH = 'amazon_food_reviews/train_embedding_sen.npy'
# DEV_SENTENCE_EMBEDDING_PATH = 'amazon_food_reviews/dev_embedding_sen.npy'
# TEST_SENTENCE_EMBEDDING_PATH = 'amazon_food_reviews/test_embedding_sen.npy'
# TRAIN_LABEL_PATH = 'amazon_food_reviews/train_label.npy'
# DEV_LABEL_PATH = 'amazon_food_reviews/dev_label.npy'
# TEST_LABEL_PATH = 'amazon_food_reviews/test_label.npy'


tripadvisor_path = 'trip advisor'
TRAIN_WORD_EMBEDDING_PATH = 'trip advisor/train_embedding_word.npy'
DEV_WORD_EMBEDDING_PATH = 'trip advisor/dev_embedding_word.npy'
TEST_WORD_EMBEDDING_PATH = 'trip advisor/test_embedding_word.npy'
TRAIN_SENTENCE_EMBEDDING_PATH = 'trip advisor/train_embedding_sen.npy'
DEV_SENTENCE_EMBEDDING_PATH = 'trip advisor/dev_embedding_sen.npy'
TEST_SENTENCE_EMBEDDING_PATH = 'trip advisor/test_embedding_sen.npy'
TRAIN_LABEL_PATH = 'trip advisor/train_label.npy'
DEV_LABEL_PATH = 'trip advisor/dev_label.npy'
TEST_LABEL_PATH = 'trip advisor/test_label.npy'


flags = tf.flags

FLAGS = flags.FLAGS


# Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")
# flags.DEFINE_string("task_name", None, "The name of the task to train.")


class Dataprocessor_trip(object):
    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file) as fin:
            lines = fin.readlines()
        return lines

    def get_train_examples(self, data_dir):
        return self._create_examples(
            # self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
            self._read_file(os.path.join(data_dir, "train")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "test")), "test")

    def get_labels(self):
        return ['0', '1', '2', '3', '4', '5']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        label_checkin = []
        label_value = []
        label_business = []
        label_service = []
        label_cleanliness = []
        label_location = []
        label_room = []
        label_overall = []

        for (i, line) in enumerate(lines):
            # print('------------------------------')
            # print('line:{}'.format(line))
            seg = line.split('\t\t')
            # print('------------------------------')
            # print('new_line:{}'.format(line))
            # 文本部分
            text_a = tokenization.convert_to_unicode(' '.join(seg[-1].split('<ssssss>')).replace('\n', '').replace('。','').replace('，','')\
                        .replace('-', ' ').replace(',', ' ').replace('@', ' ')\
                        .replace(' 1\n', '').replace(' 2\n', '').replace(' 3\n', '').replace(' 4\n', '').replace(' 5\n', '')\
                        .replace(' 7\n', '').replace(' 8\n', '').replace(' 9\n', '').replace(' 10\n', '').replace('<br />', ' ')\
                        .replace("\"", " ").replace(":", " ").replace('=', ' ').replace('[', ' ').replace(']', ' ').replace('+', ' ')\
                        .replace(';', ' ').replace('*', '').replace('_', '').replace('\'s', ' ').replace('\' ', ' ').replace('\n', ' ')\
                        .replace('~', ' ').replace('&', ' ').replace('/', ' ').replace('\"', '').replace('$', '').replace('%', ''))

            examples.append(text_a)
            # 标签部分
            tmp_labels = seg[0].split(' ')
            checkin = int(tmp_labels[-1])
            if checkin == -1:
                checkin = 0
                # 标签从0开始
            label_checkin.append(checkin)
            value = int(tmp_labels[-2])
            if value == -1:
                value = 0
            label_value.append(value)
            business = int(tmp_labels[-3])
            if business == -1:
                business = 0
            label_business.append(business)
            service = int(tmp_labels[-4])
            if service == -1:
                service = 0
            label_service.append(service)
            cleanliness = int(tmp_labels[-5])
            if cleanliness == -1:
                cleanliness = 0
            label_cleanliness.append(cleanliness)
            location = int(tmp_labels[-6])
            if location == -1:
                location = 0
            label_location.append(location)
            room = int(tmp_labels[-7])
            if room == -1:
                room = 0
            label_room.append(room)
            overall = int(tmp_labels[-8])
            if overall == -1:
                overall = 0
            label_overall.append(overall)

        labels = [label_overall,
                    label_checkin,
                    label_value,
                    label_business,
                    label_service,
                    label_cleanliness,
                    label_location,
                    label_room
                    ]

        return examples, labels


class Dataprocessor_yelp(object):
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
        return lines

    def get_train_examples(self, data_dir):
        return self._create_examples(
            # self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
            self._read_tsv(os.path.join(data_dir, "train")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test")), "test")

    def get_predict_examples(self, data_dir):
        return self._create_predict_examples(
            self._read_tsv(os.path.join(data_dir, "predict.tsv")), "predict")

    def get_labels(self):
        return ["1", "2", "3", "4", "5"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        labels = []
        for (i, line) in enumerate(lines):
            # print('------------------------------')
            # print('line:{}'.format(line))
            line = line[0].split(' ')
            # print('------------------------------')
            # print('new_line:{}'.format(line))
            text_a = tokenization.convert_to_unicode(' '.join(line[0:-1]).replace('\n', '').replace('。','').replace('，','')\
                        .replace('-', ' ').replace(',', ' ').replace('@', ' ')\
                        .replace(' 1\n', '').replace(' 2\n', '').replace(' 3\n', '').replace(' 4\n', '').replace(' 5\n', '')\
                        .replace(' 7\n', '').replace(' 8\n', '').replace(' 9\n', '').replace(' 10\n', '').replace('<br />', ' ')\
                        .replace("\"", " ").replace(":", " ").replace('=', ' ').replace('[', ' ').replace(']', ' ').replace('+', ' ')\
                        .replace(';', ' ').replace('*', '').replace('_', '').replace('\'s', ' ').replace('\' ', ' ').replace('\n', ' ')\
                        .replace('~', ' ').replace('&', ' ').replace('/', ' ').replace('\"', '').replace('$', '').replace('%', ''))
            label = int(line[-1])-1 # 标签从0开始
            examples.append(text_a)
            labels.append(label)

        return examples, labels

    def _create_predict_examples(self, lines, set_type):
        """Creates examples for the predict sets."""
        examples = []
        for (i, line) in enumerate(lines):
            line = line[0].split(' ')
            text_a = tokenization.convert_to_unicode(' '.join(line[0:-1]).replace('\n', '').replace('。','').replace('，','')\
                        .replace('-', ' ').replace(',', ' ').replace('@', ' ').replace('/', ' ')\
                        .replace(' 1\n', '').replace(' 2\n', '').replace(' 3\n', '').replace(' 4\n', '').replace(' 5\n', '')\
                        .replace(' 7\n', '').replace(' 8\n', '').replace(' 9\n', '').replace(' 10\n', '').replace('<br />', ' ')\
                        .replace("\"", " ").replace(":", " ").replace('=', ' ').replace('[', ' ').replace(']', ' ').replace('+', ' ')\
                        .replace(';', ' ').replace('*', '').replace('_', '').replace('\'s', ' ').replace('\' ', ' ').replace('\n', ' ')\
                        .replace('~', ' ').replace('&', ' ').replace('/', ' ').replace('\"', '').replace('$', '').replace('%', ''))
            examples.append(text_a)
        return examples


def data_sentence_embedding(data_set='train'):
    """
    :function: 通过bert_as_service 生成text的向量表示
    :param data_set: 数据集类型，default=train， 根据实际需要可以设置为train，dev，test, predict
    :return: data_embeddings and labels
    """
    data_getter = Dataprocessor_yelp()
    # data_getter = Dataprocessor_trip()
    if data_set == 'train':
        if os.path.exists(TRAIN_SENTENCE_EMBEDDING_PATH) and os.path.exists(TRAIN_LABEL_PATH):
            print('----Load train embeddings----')
            train_embedding = np.load(TRAIN_SENTENCE_EMBEDDING_PATH)
            train_labels = np.load(TRAIN_LABEL_PATH)

            return train_embedding, train_labels

        else:
            train_examples, train_labels = data_getter.get_train_examples(data_dir=Yelp_data_path)
            # train_examples, train_labels = data_getter.get_train_examples(data_dir="Amazon_mobile_reviews")
            print('----Generate train embeddings----')
            bc = BertClient(ip='192.168.3.4')
            train_embedding = bc.encode(train_examples)
            print('----Save train embeddings----')
            np.save(TRAIN_SENTENCE_EMBEDDING_PATH, train_embedding)
            np.save(TRAIN_LABEL_PATH, train_labels)
            print('----Save train embeddings DONE----')

            return train_embedding, train_labels


    elif data_set == 'dev':
        if os.path.exists(DEV_SENTENCE_EMBEDDING_PATH) and os.path.exists(DEV_LABEL_PATH):
            print('----Load dev embeddings----')
            dev_embedding = np.load(DEV_SENTENCE_EMBEDDING_PATH)
            dev_labels = np.load(DEV_LABEL_PATH)
            return dev_embedding, dev_labels

        else:
            dev_examples, dev_labels = data_getter.get_dev_examples(data_dir=Yelp_data_path)
            # dev_examples, dev_labels = data_getter.get_dev_examples(data_dir="Amazon_mobile_reviews")
            print('----Generate dev embeddings----')
            bc = BertClient(ip='192.168.3.4')
            dev_embedding = bc.encode(dev_examples)
            print('----Save dev embeddings----')
            np.save(DEV_SENTENCE_EMBEDDING_PATH, dev_embedding)
            np.save(DEV_LABEL_PATH, dev_labels)
            print('----Save dev embeddings DONE----')
            return dev_embedding, dev_labels

    elif data_set == 'test':
        if os.path.exists(TEST_SENTENCE_EMBEDDING_PATH) and os.path.exists(TEST_LABEL_PATH):
            print('----Load test embeddings----')
            test_embedding = np.load(TEST_SENTENCE_EMBEDDING_PATH)
            test_labels = np.load(TEST_LABEL_PATH)
            return test_embedding, test_labels

        else:
            test_examples, test_labels = data_getter.get_test_examples(data_dir=Yelp_data_path)
            # test_examples, test_labels = data_getter.get_test_examples(data_dir="Amazon_mobile_reviews")
            print('----Generate test embeddings----')
            bc = BertClient(ip='192.168.3.4')
            test_embedding = bc.encode(test_examples)
            print('----Save test embeddings----')
            np.save(TEST_SENTENCE_EMBEDDING_PATH, test_embedding)
            np.save(TEST_LABEL_PATH, test_labels)
            print('----Save test embeddings DONE----')
            return test_embedding, test_labels


def data_word_embedding(data_set='train'):
    """
    :function: 通过bert_as_service 生成text的向量表示
    :param data_set: 数据集类型，default=train， 根据实际需要可以设置为train，dev，test, predict
    :return: data_embeddings and labels
    """
    # data_getter = Dataprocessor_yelp()
    data_getter = Dataprocessor_trip()
    if data_set == 'train':
        if os.path.exists(TRAIN_WORD_EMBEDDING_PATH) and os.path.exists(TRAIN_LABEL_PATH):
            print('----Load train embeddings----')
            train_embedding = np.load(TRAIN_WORD_EMBEDDING_PATH)
            train_labels = np.load(TRAIN_LABEL_PATH)
            # train, train_labels = data_getter.get_train_examples(data_dir=tripadvisor_path)
            # np.save(TRAIN_LABEL_PATH, train_labels)
            return train_embedding, train_labels


        else:
            train_examples, train_labels = data_getter.get_train_examples(data_dir=tripadvisor_path)
            # train_examples, train_labels = data_getter.get_train_examples(data_dir="Amazon_mobile_reviews")
            print('----Generate train embeddings----')
            bc = BertClient(ip='192.168.3.4')
            train_embedding = []
            for (i, item) in enumerate(train_examples):
                print("train {}".format(i))
                train_embedding.append(bc.encode([item])[0])
            print('----Save train embeddings----')
            np.save(TRAIN_WORD_EMBEDDING_PATH, train_embedding)
            np.save(TRAIN_LABEL_PATH, train_labels)
            print('----Save train embeddings DONE----')

            return train_embedding, train_labels

    elif data_set == 'dev':
        if os.path.exists(DEV_WORD_EMBEDDING_PATH) and os.path.exists(DEV_LABEL_PATH):
            print('----Load dev embeddings----')
            dev_embedding = np.load(DEV_WORD_EMBEDDING_PATH)
            dev_labels = np.load(DEV_LABEL_PATH)
            # train, dev_labels = data_getter.get_dev_examples(data_dir=tripadvisor_path)
            # np.save(DEV_LABEL_PATH, dev_labels)
            return dev_embedding, dev_labels

        else:
            dev_examples, dev_labels = data_getter.get_dev_examples(data_dir=tripadvisor_path)
            # dev_examples, dev_labels = data_getter.get_dev_examples(data_dir="Amazon_mobile_reviews")
            print('----Generate dev embeddings----')
            bc = BertClient(ip='192.168.3.4')
            dev_embedding = bc.encode(dev_examples)
            print('----Save dev embeddings----')
            np.save(DEV_WORD_EMBEDDING_PATH, dev_embedding)
            np.save(DEV_LABEL_PATH, dev_labels)
            print('----Save dev embeddings DONE----')
            return dev_embedding, dev_labels

    elif data_set == 'test':
        if os.path.exists(TEST_WORD_EMBEDDING_PATH) and os.path.exists(TEST_LABEL_PATH):
            print('----Load test embeddings----')
            test_embedding = np.load(TEST_WORD_EMBEDDING_PATH)
            test_labels = np.load(TEST_LABEL_PATH)
            # train, test_labels = data_getter.get_test_examples(data_dir=tripadvisor_path)
            # np.save(TEST_LABEL_PATH, test_labels)
            return test_embedding, test_labels

        else:
            test_examples, test_labels = data_getter.get_test_examples(data_dir=tripadvisor_path)
            # test_examples, test_labels = data_getter.get_test_examples(data_dir="Amazon_mobile_reviews")
            print('----Generate test embeddings----')
            bc = BertClient(ip='192.168.3.4')
            test_embedding = []
            for (i, item) in enumerate(test_examples):
                print("test {}".format(i))
                test_embedding.append(bc.encode([item])[0])
            print('----Save test embeddings----')
            np.save(TEST_WORD_EMBEDDING_PATH, test_embedding)
            np.save(TEST_LABEL_PATH, test_labels)
            print('----Save test embeddings DONE----')
            return test_embedding, test_labels

    elif data_set == 'predict':
        predict_examples = data_getter.get_test_examples(data_dir=tripadvisor_path)
        bc = BertClient(ip='192.168.3.4')
        # flags.MAX_predict_length = max(predict_examples, key=len)
        predict_embedding = bc.encode(predict_examples)
        return predict_embedding


def changenumberintovector(array):
    result = []
    for item in array:
        if item == 0:
            result.append([1, 0, 0, 0, 0])
        elif item == 1:
            result.append([0, 1, 0, 0, 0])
        elif item == 2:
            result.append([0, 0, 1, 0, 0])
        elif item == 3:
            result.append([0, 0, 0, 1, 0])
        elif item == 4:
            result.append([0, 0, 0, 0, 1])
    return result


if __name__ == '__main__':
    # data_getter = ReadDataFromFile()
    # bc = BertClient(ip='192.168.3.4')
    # train_examples, train_labels = data_getter.get_train_examples(data_dir="yelp_new_reviews")
    # train_embedding = bc.encode(train_examples)
    # print('-------------------')
    # print('shape:', train_embedding.shape)
    # print('train_embedding', train_embedding)
    # print('000000000000000000000000000')
    # train_x, train_y = data_word_embedding('train')
    test_x, test_y = data_word_embedding('test')
    print('111111')
    #
    # # test_x, test_y = data_helper.data_embedding('test')
    # bc = BertClient(ip='192.168.3.4')
    # embedding_data = bc.encode([
    #     'a good place',
    #     'a great place',
        # 'I hadn\'t intended on coming here',
        # 'because there was no free wifi.',
        # 'the menu was student-budget friendly.',
        # 'The scone was big but not particularly cheesy.',
        # 'a Hot sunny Friday afternoon with no wait.',
        # 'Slow service, bored staff, mediocre food and an over all unpleasant experience.',
        # 'These are not good people the manger Johnathan smokes weed. Most of the employees are on drugs.',
        # 'The service is slow.',
        # 'Beautiful cozy place to relax with a snack and cup of coffee.',
        # 'Love this place! ',
        # 'Like this place! ',
        # 'Great food and healthy too.',
        # 'A great variety of choices on the menu.',
        # 'The fish tastes great!'
        # 'Usually this bar is fun and friendly, however not on sunday nights apparently!',
        # 'If I could give this place no stars, I would be happier.'
    # ])
    # a = embedding_data
    # print(a.shape)
    # print(a)
    # a.tofile(file='testembedding.tsv', sep='\t', format='%s')
    # # b = (np.fromfile(file='testembedding.tsv', dtype=np.float, count=-1, sep=',').reshape((100, 120, 768)))
    # with open('embed', 'w', encoding='utf-8') as fp:
    #     fp.write('good\t')
    #     fp.write(','.join(str(v) for v in a[0][2]))
    #     fp.write('\n')
    #     fp.write('great\t')
    #     fp.write(','.join(str(v) for v in a[1][2]))
    #     fp.write('\n')
    #
    #     fp.write('place\t')
    #     fp.write(','.join(str(v) for v in a[0][3]))
    #     fp.write('\n')
    #     fp.write('place\t')
    #     fp.write(','.join(str(v) for v in a[1][3]))
    #     fp.write('\n')
    #     fp.write('place\t')
    #     fp.write(','.join(str(v) for v in a[2][3]))
    #     fp.write('\n')
    #     fp.write('place\t')
    #     fp.write(','.join(str(v) for v in a[3][3]))
    #     fp.write('\n')
    #
    #     fp.write('Love\t')
    #     fp.write(','.join(str(v) for v in a[2][1]))
    #     fp.write('\n')
    #     fp.write('Like\t')
    #     fp.write(','.join(str(v) for v in a[3][1]))
    #     fp.write('\n')
    #
    #     fp.write('Great\t')
    #     fp.write(','.join(str(v) for v in a[4][1]))
    #     fp.write('\n')
    #     fp.write('great\t')
    #     fp.write(','.join(str(v) for v in a[5][2]))
    #     fp.write('\n')
    #     fp.write('great\t')
    #     fp.write(','.join(str(v) for v in a[6][4]))
    #     fp.write('\n')

    # bc = BertClient(ip='192.168.3.4')
    # embedding_data = bc.encode(
    #     [
    #         'hello world! ||| I\'m comming'
    #     ], show_tokens=True)
    # # np.save('test.npy', embedding_data)
    # print('111111111')
