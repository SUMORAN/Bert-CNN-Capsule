# -*- encoding: utf-8 -*-

import json
import numpy as np
from math import log
import linecache
import codecs
import random
import csv
import matplotlib.pyplot as plt


# 每个yelp评论文件包含的评论数
yelp_sample_num = 50000
amazon_sample_num = 50000


# 加载词向量，通过参数控制加载的数量
def load_word_ebd(model_path, loading_num=500000):
    with open(model_path, 'r', encoding='utf-8') as f:
        i = 0
        dict = {}
        for line in f:
            word = line.split()[0]
            vec = np.fromstring(line.replace(word, '').replace('\n', ''), dtype='float32', sep=' ')
            if i <= loading_num:
                if word.lower() not in dict:
                    dict[word.lower()] = vec
                    i += 1
            else:
                break
            print("Loading word vector. Finished word:"+str(i))
    dict['ph'] = np.zeros(300)
    return dict


# 加载一些否定词，避免在使用LDA表示句子时忽略掉否定词
def load_neg_words(neg_words_path):
    neg_words = []
    with open(neg_words_path, 'r', encoding='utf-8') as f:
        for line in f:
            neg_words.append(line.replace('\n', ''))
    return neg_words


# 将词语加入对应的字典中，并增加次数
def add_word_to_dict(word, dict):
    word = word.lower()
    if word in dict:
        dict[word] += 1
    else:
        dict[word] = 1


# 移除每行文本中的无用词语和标点
def rm_useless_tokens(line):
    new_line = line.replace('.', ' ').replace('(', ' ').replace(')', ' ')\
        .replace('-', ' ').replace(',', ' ').replace('@', ' ')\
        .replace('1\n', '').replace('2\n', '').replace('3\n', '').replace('4\n', '').replace('5\n', '').replace('0\n', '')\
        .replace('7\n', '').replace('8\n', '').replace('9\n', '').replace('10\n', '').replace('<br />', ' ')\
        .replace("\"", " ").replace(":", " ").replace('=', ' ').replace('[', ' ').replace(']', ' ').replace('+', ' ')\
        .replace(';', ' ').replace('*', '').replace('_', '').replace('\'s', ' ').replace('\' ', ' ').replace('\n', ' ')\
        .replace('~', ' ').replace('&', ' ').replace('/', ' ').replace('\"', '').replace('$', '').replace('%', '').replace('\t', '')
    return new_line


# 移除文本蕴含数据中的每一行中无用标点和词语
def rm_useless_tokens_for_etm(line):
    new_line = line.replace('(', ' ').replace(')', ' ') \
        .replace('?', ' ').replace('!', ' ').replace('-', ' ').replace('/', ' ').replace(',', ' ').replace('@', ' ') \
        .replace(' 0\n', '').replace(' 1\n', '').replace(' 2\n', '').replace(' 3\n', '').replace(' 4\n', '').replace(' 5\n', '') \
        .replace("\"", " ").replace(":", " ").replace('=', ' ').replace('[', ' ').replace(']', ' ').replace('+', ' ') \
        .replace(';', ' ').replace('*', '').replace('_', '').replace('\'s', ' ').replace('\' ', ' ').replace('\n', ' ') \
        .replace('~', ' ').replace('&', ' ')
    return new_line


# 去除停用词
def remove_stop_words(one_line, stop_words):
    for stop_word in stop_words:
        if one_line.split(" ")[0] == stop_word:
            one_line = one_line.replace(stop_word+" ", "")
        else:
            one_line = one_line.replace(" "+stop_word+" ", " ")

    return one_line


# 根据出现次数，移除字典中出现次数较少的词语
def rm_useless_tokens_in_dict(dict):
    avg_count = sum(dict.values())/len(dict)
    new_dict = {}
    for key in dict:
        if dict[key] >= avg_count:
            new_dict[key] = dict[key]
    return new_dict


# 统计在所有类别中出现词语的词频
def cal_word_freq_for_all_cat(review_path):
    word_freq_dict1 = {}
    word_freq_dict2 = {}
    word_freq_dict3 = {}
    word_freq_dict4 = {}
    word_freq_dict5 = {}
    with open(review_path, 'r') as f:
        i = 0
        for line in f:
            new_line = rm_useless_tokens(line)
            words = new_line.split()
            for word in words:
                if ' 1\n' in line:
                    add_word_to_dict(word, word_freq_dict1)
                elif ' 2\n' in line:
                    add_word_to_dict(word, word_freq_dict2)
                elif ' 3\n' in line:
                    add_word_to_dict(word, word_freq_dict3)
                elif ' 4\n' in line:
                    add_word_to_dict(word, word_freq_dict4)
                else:
                    add_word_to_dict(word, word_freq_dict5)
            print('Cal word freq. Finished sentence:' + str(i))
            i += 1

        word_freq_dict1 = rm_useless_tokens_in_dict(word_freq_dict1)
        word_freq_dict2 = rm_useless_tokens_in_dict(word_freq_dict2)
        word_freq_dict3 = rm_useless_tokens_in_dict(word_freq_dict3)
        word_freq_dict4 = rm_useless_tokens_in_dict(word_freq_dict4)
        word_freq_dict5 = rm_useless_tokens_in_dict(word_freq_dict5)
    return word_freq_dict1, word_freq_dict2, word_freq_dict3, word_freq_dict4, word_freq_dict5


# 统计一个文件中各类别中各词语的频数
def cal_word_freq_for_doc(review_path, read_num=50000):
    stop_words = load_stop_words("yelp_glove/stop_words_1")  # 加载停用词
    word_freq_dict1 = {}
    word_freq_dict2 = {}
    word_freq_dict3 = {}
    word_freq_dict4 = {}
    word_freq_dict5 = {}
    i = 0
    with open(review_path, 'r', encoding='utf-8') as f:
        for line in f:
            new_line = rm_useless_tokens(line.lower())  # 去除无用的符号
            new_line = remove_stop_words(new_line, stop_words)  # 去除停用词
            words = new_line.split()
            for word in words:
                if ' 1\n' in line:
                    add_word_to_dict(word, word_freq_dict1)
                elif ' 2\n' in line:
                    add_word_to_dict(word, word_freq_dict2)
                elif ' 3\n' in line:
                    add_word_to_dict(word, word_freq_dict3)
                elif ' 4\n' in line:
                    add_word_to_dict(word, word_freq_dict4)
                else:
                    add_word_to_dict(word, word_freq_dict5)
            print('Cal word freq. Finished sentence:' + str(i))
            i += 1
            if i > read_num:
                break

    word_freq_dict1 = rm_useless_tokens_in_dict(word_freq_dict1)
    word_freq_dict2 = rm_useless_tokens_in_dict(word_freq_dict2)
    word_freq_dict3 = rm_useless_tokens_in_dict(word_freq_dict3)
    word_freq_dict4 = rm_useless_tokens_in_dict(word_freq_dict4)
    word_freq_dict5 = rm_useless_tokens_in_dict(word_freq_dict5)
    return word_freq_dict1, word_freq_dict2, word_freq_dict3, word_freq_dict4, word_freq_dict5


# 加载停用词
def load_stop_words(stopwords_path):
    stopwords = []
    with open(stopwords_path, 'r') as f:
        for line in f:
            stopwords.append(line.replace('\n', ''))
    return stopwords


# 划分训练集与测试集（对yelp评论数据集）
def split_train_test_for_yelp(review_paths):
    train_path = review_paths[0]
    test_path = review_paths[1]

    all_sents = []

    train_x = []
    train_y = []
    test_x = []
    test_y = []

    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(all_sents) < 45000:
                all_sents.append(line)

    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(all_sents) < 47500:
                all_sents.append(line)

    test_sent = all_sents[45000:]
    train_sent = all_sents[:45000]

    for line in train_sent:
        if ' 1\n' in line:
            train_y.append(0)
        elif ' 2\n' in line:
            train_y.append(1)
        elif ' 3\n' in line:
            train_y.append(2)
        elif ' 4\n' in line:
            train_y.append(3)
        else:
            train_y.append(4)
        new_line = rm_useless_tokens(line.lower())  # 移除无用符号
        train_x.append(new_line)
    for line in test_sent:
        if ' 1\n' in line:
            test_y.append(0)
        elif ' 2\n' in line:
            test_y.append(1)
        elif ' 3\n' in line:
            test_y.append(2)
        elif ' 4\n' in line:
            test_y.append(3)
        else:
            test_y.append(4)
        new_line = rm_useless_tokens(line.lower())  # 移除无用符号
        test_x.append(new_line)

    return train_x, train_y, test_x, test_y


# 划分训练集与测试集（对于amazon食品评论数据集）
def split_train_test_for_amazon(review_path, train_size=0.9):
    train_num = amazon_sample_num * train_size
    train_x = []
    train_smr = []
    train_y = []
    dev_x = []
    dev_smr = []
    dev_y = []
    test_x = []
    test_smr = []
    test_y = []

    with open("Amazon_reviews/train_smr", 'r', encoding='utf-8') as f:
        for line in f:
            x = line.replace(' 1-*-', '-*-')\
                .replace(' 2-*-', '-*-').replace(' 3-*-', '-*-').replace(' 4-*-', '-*-').replace(' 5-*-', '-*-')
            if ' 1-*-' in line:
                train_y.append(0)
            elif ' 2-*-' in line:
                train_y.append(1)
            elif ' 3-*-' in line:
                train_y.append(2)
            elif ' 4-*-' in line:
                train_y.append(3)
            elif ' 5-*-' in line:
                train_y.append(4)
            text = x.split('-*-')[0]
            summary = x.split('-*-')[1]
            new_text = rm_useless_tokens(text.lower())  # 移除无用符号
            new_smr = rm_useless_tokens(summary.lower())
            train_x.append(new_text)
            train_smr.append(new_smr)

    with open("Amazon_reviews/test_smr", 'r', encoding='utf-8') as f:
        for line in f:
            x = line.replace(' 1-*-', '-*-') \
                .replace(' 2-*-', '-*-').replace(' 3-*-', '-*-').replace(' 4-*-', '-*-').replace(' 5-*-', '-*-')
            if ' 1-*-' in line:
                test_y.append(0)
            elif ' 2-*-' in line:
                test_y.append(1)
            elif ' 3-*-' in line:
                test_y.append(2)
            elif ' 4-*-' in line:
                test_y.append(3)
            elif ' 5-*-' in line:
                test_y.append(4)
            text = x.split('-*-')[0]
            summary = x.split('-*-')[1]
            new_text = rm_useless_tokens(text.lower())  # 移除无用符号
            new_smr = rm_useless_tokens(summary.lower())
            test_x.append(new_text)
            test_smr.append(new_smr)

    with open("Amazon_reviews/dev_smr", 'r', encoding='utf-8') as f:
        for line in f:
            x = line.replace(' 1-*-', '-*-') \
                .replace(' 2-*-', '-*-').replace(' 3-*-', '-*-').replace(' 4-*-', '-*-').replace(' 5-*-', '-*-')
            if ' 1-*-' in line:
                dev_y.append(0)
            elif ' 2-*-' in line:
                dev_y.append(1)
            elif ' 3-*-' in line:
                dev_y.append(2)
            elif ' 4-*-' in line:
                dev_y.append(3)
            elif ' 5-*-' in line:
                dev_y.append(4)
            text = x.split('-*-')[0]
            summary = x.split('-*-')[1]
            new_text = rm_useless_tokens(text.lower())  # 移除无用符号
            new_smr = rm_useless_tokens(summary.lower())
            dev_x.append(new_text)
            dev_smr.append(new_smr)

    return train_x, train_smr, train_y, \
           dev_x, dev_smr, dev_y, \
           test_x, test_smr, test_y


# 读取双倍数据量的亚马逊食品评论训练集和测试集
def split_train_test_for_db_amazon(review_paths):
    train_path = review_paths[0]
    test_path = review_paths[1]

    all_sents = []

    train_x = []
    train_y = []
    test_x = []
    test_y = []

    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            all_sents.append(line)

    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            all_sents.append(line)

    test_sent = all_sents[45000:]
    train_sent = all_sents[:45000]

    for line in train_sent:
        if ' 1\n' in line:
            train_y.append(0)
        elif ' 2\n' in line:
            train_y.append(1)
        elif ' 3\n' in line:
            train_y.append(2)
        elif ' 4\n' in line:
            train_y.append(3)
        else:
            train_y.append(4)
        new_line = rm_useless_tokens(line.lower())  # 移除无用符号
        train_x.append(new_line)
    for line in test_sent:
        if ' 1\n' in line:
            test_y.append(0)
        elif ' 2\n' in line:
            test_y.append(1)
        elif ' 3\n' in line:
            test_y.append(2)
        elif ' 4\n' in line:
            test_y.append(3)
        else:
            test_y.append(4)
        new_line = rm_useless_tokens(line.lower())  # 移除无用符号
        test_x.append(new_line)

    return train_x, train_y, test_x, test_y


# 划分训练集和测试集（对于amazon电子产品评论数据集）
def split_train_test_for_mobile(review_paths):
    train_path = review_paths[0]
    test_path = review_paths[1]
    dev_path = review_paths[2]

    all_sents = []

    train_x = []
    train_y = []
    test_x = []
    test_y = []
    dev_x = []
    dev_y = []

    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            all_sents.append(line)

    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            all_sents.append(line)

    test_sent = all_sents[45000:]
    train_sent = all_sents[:45000]

    for line in train_sent:
        if ' 1\n' in line:
            train_y.append(0)
        elif ' 2\n' in line:
            train_y.append(1)
        elif ' 3\n' in line:
            train_y.append(2)
        elif ' 4\n' in line:
            train_y.append(3)
        else:
            train_y.append(4)
        new_line = rm_useless_tokens(line.lower())  # 移除无用符号
        train_x.append(new_line)
    for line in test_sent:
        if ' 1\n' in line:
            test_y.append(0)
        elif ' 2\n' in line:
            test_y.append(1)
        elif ' 3\n' in line:
            test_y.append(2)
        elif ' 4\n' in line:
            test_y.append(3)
        else:
            test_y.append(4)
        new_line = rm_useless_tokens(line.lower())  # 移除无用符号
        test_x.append(new_line)

    with open(dev_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ' 1\n' in line:
                dev_y.append(0)
            elif ' 2\n' in line:
                dev_y.append(1)
            elif ' 3\n' in line:
                dev_y.append(2)
            elif ' 4\n' in line:
                dev_y.append(3)
            else:
                dev_y.append(4)
            new_line = rm_useless_tokens(line.lower())  # 移除无用符号
            #new_line = trunc_sent_by_conj(line.lower())
            dev_x.append(new_line)

    return train_x, train_y, test_x, test_y, dev_x, dev_y



def split_train_test_for_zhang(review_paths):
    train_path = review_paths[0]
    test_path = review_paths[1]

    train_x = []
    train_y = []
    test_x = []
    test_y = []

    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '1\n' in line:
                train_y.append(0)
            elif '2\n' in line:
                train_y.append(1)
            elif '3\n' in line:
                train_y.append(2)
            elif '4\n' in line:
                train_y.append(3)
            elif '5\n' in line:
                train_y.append(4)

            new_line = rm_useless_tokens(line.lower())  # 移除无用符号
            train_x.append(new_line)

    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '1\n' in line:
                test_y.append(0)
            elif '2\n' in line:
                test_y.append(1)
            elif '3\n' in line:
                test_y.append(2)
            elif '4\n' in line:
                test_y.append(3)
            elif '5\n' in line:
                test_y.append(4)
            new_line = rm_useless_tokens(line.lower())  # 移除无用符号
            test_x.append(new_line)

    return train_x, train_y, test_x, test_y


# 根据一些连词截取句子中重要部分
def trunc_sent_by_conj(sent):
    new_sent = rm_useless_tokens(sent.lower())
    words = new_sent.split()
    if 'but' in words:
        return ' '.join(words[words.index('but')+1:])
    else:
        return new_sent


# 划分训练集与测试集（对产业链数据）
def split_train_test_for_ic(ic_paths, train_size=0.8):
    train_x = []
    train_y = []
    dev_x = []
    dev_y = []
    test_x = []
    test_y = []

    for path in ic_paths:
        i = 0
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            line_nums = len(lines)
            train_num = line_nums * train_size
            for line in lines:
                if i < train_num:  # 训练集
                    if ' 1\n' in line:
                        train_y.append(1)
                    elif ' 0\n' in line:
                        train_y.append(0)
                    train_x.append(line.replace(' 1\n', '').replace(' 0\n', ''))
                elif train_num <= i < line_nums*(train_size + (1-train_size)/2):  # 测试集
                    if ' 1\n' in line:
                        test_y.append(1)
                    elif ' 0\n' in line:
                        test_y.append(0)
                    test_x.append(line.replace(' 1\n', '').replace(' 0\n', ''))
                elif line_nums*(train_size + (1-train_size)/2) <= i < line_nums:  # 验证集
                    if ' 1\n' in line:
                        dev_y.append(1)
                    elif ' 0\n' in line:
                        dev_y.append(1)
                    dev_x.append(line.replace(' 1\n', '').replace(' 0\n', ''))
                i += 1
                print("Splitting sentence:"+str(i))
    return train_x, train_y, dev_x, dev_y, test_x, test_y


# 读取文本蕴含数据
def split_train_test_for_text_etm(data_path, train_size=0.9):
    lines = linecache.getlines(data_path)
    train_num = len(lines) * train_size

    train_x = []
    train_y = []
    dev_x = []
    dev_y = []
    test_x = []
    test_y = []

    for i in range(0, len(lines)):
        line = lines[i]
        if i < train_num:
            if ' 0\n' in line:
                train_y.append(0)
            elif ' 1\n' in line:
                train_y.append(1)
            elif ' 2\n' in line:
                train_y.append(2)
            new_line = rm_useless_tokens_for_etm(line.lower())  # 移除无用符号
            train_x.append(new_line)
        elif train_num <= i < len(lines) * ((1-train_size)/2+train_size):
            if ' 0\n' in line:  # 测试集
                test_y.append(0)
            elif ' 1\n' in line:
                test_y.append(1)
            elif ' 2\n' in line:
                test_y.append(2)
            new_line = rm_useless_tokens_for_etm(line.lower())
            test_x.append(new_line)
        elif len(lines) * ((1-train_size)/2+train_size) <= i < len(lines):
            if ' 0\n' in line:  # 验证集
                dev_y.append(0)
            elif ' 1\n' in line:
                dev_y.append(1)
            elif ' 2\n' in line:
                dev_y.append(2)
            new_line = rm_useless_tokens_for_etm(line.lower())
            dev_x.append(new_line)
        print("Finished loading sentence:"+str(i))
    return train_x, train_y, dev_x, dev_y, test_x, test_y


# 读取imdb影评数据（训练集和测试集已经被划分好，直接读取即可）
def read_imdb_data(data_path):
    x = []
    y = []
    lines = linecache.getlines(data_path)
    for line in lines:
        if ' 1\n' in line:
            y.append(0)
        elif ' 2\n' in line:
            y.append(1)
        elif ' 3\n' in line:
            y.append(2)
        elif ' 4\n' in line:
            y.append(3)
        elif ' 7\n' in line:
            y.append(4)
        elif ' 8\n' in line:
            y.append(5)
        elif ' 9\n' in line:
            y.append(6)
        elif ' 10\n' in line:
            y.append(7)
        new_line = rm_useless_tokens(line.lower())
        x.append(new_line)
    return x, y


def read_dbpedia_data(data_path):
    x = []
    y = []
    lines = linecache.getlines(data_path)
    for line in lines:
        if ' 1\n' in line:
            y.append(0)
        elif ' 2\n' in line:
            y.append(1)
        elif ' 3\n' in line:
            y.append(2)
        elif ' 4\n' in line:
            y.append(3)
        elif ' 5\n' in line:
            y.append(4)
        elif ' 6\n' in line:
            y.append(5)
        elif ' 7\n' in line:
            y.append(6)
        elif ' 8\n' in line:
            y.append(7)
        elif ' 9\n' in line:
            y.append(8)
        elif ' 10\n' in line:
            y.append(9)
        elif ' 11\n' in line:
            y.append(10)
        elif ' 12\n' in line:
            y.append(11)
        elif ' 13\n' in line:
            y.append(12)
        elif ' 14\n' in line:
            y.append(13)
        new_line = rm_useless_tokens(line.lower())
        x.append(new_line)
    return x, y


# 根据词语在各类样本中的频数计算其权重
def cal_word_weight(word, dict1, dict2, dict3, dict4, dict5):
    freq1 = 0
    freq2 = 0
    freq3 = 0
    freq4 = 0
    freq5 = 0
    if word in dict1:
        freq1 = dict1[word]
    if word in dict2:
        freq2 = dict2[word]
    if word in dict3:
        freq3 = dict3[word]
    if word in dict4:
        freq4 = dict4[word]
    if word in dict5:
        freq5 = dict5[word]
    temp = (freq5+freq4+1.0)/(freq3+freq2+freq1+1.0)
    return log(temp)


# 处理Amazon的CSV格式数据
# 将CSV文件写入到txt中方便操作
def write_csv_to_txt(csv_path):
    samples = []

    csv_file = csv.reader(open(csv_path, 'r'))
    i = 0

    cat1 = 0
    cat2 = 0
    cat3 = 0
    cat4 = 0
    cat5 = 0

    for review in csv_file:
        if i == 0:
            i += 1
            continue
        x = review[9]
        y = review[6]
        if len(x.split(' ')) < 5:  # 去除过短的评论
            continue
        if y == '1' and cat1 < 20000:
            samples.append(x+' '+y+'\n')
            cat1 += 1
        elif y == '2' and cat2 < 20000:
            samples.append(x + ' ' + y + '\n')
            cat2 += 1
        elif y == '3' and cat3 < 20000:
            samples.append(x + ' ' + y + '\n')
            cat3 += 1
        elif y == '4' and cat4 < 20000:
            samples.append(x + ' ' + y + '\n')
            cat4 += 1
        elif y == '5' and cat5 < 20000:
            samples.append(x + ' ' + y + '\n')
            cat5 += 1
        elif cat1 >= 20000 and cat2 >= 20000 and cat3 >= 20000 and cat4 >= 20000 and cat5 >= 20000:
            break
        i += 1
        print(i)

    random.shuffle(samples)
    train = open('Amazon_reviews/train_double', 'w', encoding='utf-8')
    test = open('Amazon_reviews/test_double', 'w', encoding='utf-8')
    i = 0
    for sample in samples:
        if i < 95000:
            train.write(sample)
        elif i >= 95000:
            test.write(sample)
        i += 1


# 将DBPedia csv文件写入txt
def write_dbpedia_csv_to_txt(csv_path):
    samples = []

    csv_file = csv.reader(open(csv_path, 'r', encoding='utf-8'))
    i = 0

    cat1 = 0
    cat2 = 0
    cat3 = 0
    cat4 = 0
    cat5 = 0
    cat6 = 0
    cat7 = 0
    cat8 = 0
    cat9 = 0
    cat10 = 0
    cat11 = 0
    cat12 = 0
    cat13 = 0
    cat14 = 0

    for review in csv_file:
        x = review[2]
        y = review[0]
        if y == '1' and cat1 < 10000:
            samples.append(x+' '+y+'\n')
            cat1 += 1
        elif y == '2' and cat2 < 10000:
            samples.append(x + ' ' + y + '\n')
            cat2 += 1
        elif y == '3' and cat3 < 10000:
            samples.append(x + ' ' + y + '\n')
            cat3 += 1
        elif y == '4' and cat4 < 10000:
            samples.append(x + ' ' + y + '\n')
            cat4 += 1
        elif y == '5' and cat5 < 10000:
            samples.append(x + ' ' + y + '\n')
            cat5 += 1
        elif y == '6' and cat6 < 10000:
            samples.append(x + ' ' + y + '\n')
            cat6 += 1
        elif y == '7' and cat7 < 10000:
            samples.append(x + ' ' + y + '\n')
            cat7 += 1
        elif y == '8' and cat8 < 10000:
            samples.append(x + ' ' + y + '\n')
            cat8 += 1
        elif y == '9' and cat9 < 10000:
            samples.append(x + ' ' + y + '\n')
            cat9 += 1
        elif y == '10' and cat10 < 10000:
            samples.append(x + ' ' + y + '\n')
            cat10 += 1
        elif y == '11' and cat11 < 10000:
            samples.append(x + ' ' + y + '\n')
            cat11 += 1
        elif y == '12' and cat12 < 10000:
            samples.append(x + ' ' + y + '\n')
            cat12 += 1
        elif y == '13' and cat13 < 10000:
            samples.append(x + ' ' + y + '\n')
            cat13 += 1
        elif y == '14' and cat14 < 10000:
            samples.append(x + ' ' + y + '\n')
            cat14 += 1
        elif cat14 >= 10000:
            break
        i += 1
        print(i)
    print()

    random.shuffle(samples)
    train = open('DBPedia/train', 'w', encoding='utf-8')
    test = open('DBPedia/test', 'w', encoding='utf-8')
    i = 0
    for sample in samples:
        if i < 70000:
            train.write(sample)
        elif 70000 < i <= 77000:
            test.write(sample)
        i += 1


def write_yahoo_csv_to_txt(csv_path):
    samples = []

    csv_file = csv.reader(open(csv_path, 'r', encoding='utf-8'))
    i = 0

    cat1 = 0
    cat2 = 0
    cat3 = 0
    cat4 = 0
    cat5 = 0
    cat6 = 0
    cat7 = 0
    cat8 = 0
    cat9 = 0
    cat10 = 0

    for review in csv_file:
        ques = review[1]
        hint = review[2]
        ans = review[3]
        y = review[0]
        if y == '1' and cat1 < 10000:
            samples.append(ques + ' ' + hint + ' ' + ans + ' ' + y + '\n')
            cat1 += 1
        elif y == '2' and cat2 < 10000:
            samples.append(ques + ' ' + hint + ' ' + ans + ' ' + y + '\n')
            cat2 += 1
        elif y == '3' and cat3 < 10000:
            samples.append(ques + ' ' + hint + ' ' + ans + ' ' + y + '\n')
            cat3 += 1
        elif y == '4' and cat4 < 10000:
            samples.append(ques + ' ' + hint + ' ' + ans + ' ' + y + '\n')
            cat4 += 1
        elif y == '5' and cat5 < 10000:
            samples.append(ques + ' ' + hint + ' ' + ans + ' ' + y + '\n')
            cat5 += 1
        elif y == '6' and cat6 < 10000:
            samples.append(ques + ' ' + hint + ' ' + ans + ' ' + y + '\n')
            cat6 += 1
        elif y == '7' and cat7 < 10000:
            samples.append(ques + ' ' + hint + ' ' + ans + ' ' + y + '\n')
            cat7 += 1
        elif y == '8' and cat8 < 10000:
            samples.append(ques + ' ' + hint + ' ' + ans + ' ' + y + '\n')
            cat8 += 1
        elif y == '9' and cat9 < 10000:
            samples.append(ques + ' ' + hint + ' ' + ans + ' ' + y + '\n')
            cat9 += 1
        elif y == '10' and cat10 < 10000:
            samples.append(ques + ' ' + hint + ' ' + ans + ' ' + y + '\n')
            cat10 += 1
        elif cat10 >= 10000:
            break
        i += 1
        print(i)
    print()

    random.shuffle(samples)
    train = open('Yahoo/train', 'w', encoding='utf-8')
    test = open('Yahoo/test', 'w', encoding='utf-8')
    i = 0
    for sample in samples:
        if i < 50000:
            train.write(sample)
        elif 50000 < i <= 55000:
            test.write(sample)
        i += 1


# 计算一个文件的评论总数
def cal_review_num(review_path):
    sum = 0
    with open(review_path, 'r', encoding='utf-8') as f:
        for line in f:
            sum += 1
            print(sum)


# 将yelp训练集按标签拆分成若干类
def split_yelp_train_data_to_cats(train_x, train_y):
    cat1_reviews = []
    cat2_reviews = []
    cat3_reviews = []
    cat4_reviews = []
    cat5_reviews = []
    for i in range(0, len(train_x)):
        x = train_x[i]
        y = train_y[i]
        if y == 0:
            cat1_reviews.append(x)
        elif y == 1:
            cat2_reviews.append(x)
        elif y == 2:
            cat3_reviews.append(x)
        elif y == 3:
            cat4_reviews.append(x)
        else:
            cat5_reviews.append(x)
        print("Finish split training data into cats:"+str(i))
    return cat1_reviews, cat2_reviews, cat3_reviews, cat4_reviews, cat5_reviews


# 将imdb训练集按标签拆分成若干类
def split_imdb_train_data_to_cats(train_x, train_y):
    cat1_reviews = []
    cat2_reviews = []
    cat3_reviews = []
    cat4_reviews = []
    cat5_reviews = []
    cat6_reviews = []
    cat7_reviews = []
    cat8_reviews = []
    for i in range(0, len(train_x)):
        x = train_x[i]
        y = train_y[i]
        if y == 0:
            cat1_reviews.append(x)
        elif y == 1:
            cat2_reviews.append(x)
        elif y == 2:
            cat3_reviews.append(x)
        elif y == 3:
            cat4_reviews.append(x)
        elif y == 4:
            cat5_reviews.append(x)
        elif y == 5:
            cat6_reviews.append(x)
        elif y == 6:
            cat7_reviews.append(x)
        elif y == 7:
            cat8_reviews.append(x)
        print("Finish split training data into cats:" + str(i))
    return cat1_reviews, cat2_reviews, cat3_reviews, cat4_reviews, cat5_reviews, cat6_reviews, cat7_reviews, cat8_reviews


# 判断一条评论中包含多少LDA关键词并返回这些关键词组成的列表
def get_contained_keywords(review, keywords, neg_words=None):
    contained_keywords = []
    words = review.split()
    if neg_words is None:
        neg_words = []

    for word in words:
        if word in keywords and word not in contained_keywords:  # 保证该词在关键词中
            contained_keywords.append(word)
    for word in words:
        if word in neg_words and word not in contained_keywords:  # 保证否定词在LDA表示中
            contained_keywords.append(word)
    return contained_keywords


# 使用LDA中得到的关键词组成的字符串来表示句子
def represent_review_with_kw(ori_x, ori_y, keywords, neg_words):
    train_ori_x = []
    train_kw_x = []
    train_kw_y = []
    for i in range(0, len(ori_x)):
        review = ori_x[i]
        contained_keywords = get_contained_keywords(review, keywords,)
        if len(contained_keywords) != 0:
            train_kw_x.append(' '.join(contained_keywords))
            train_kw_y.append(ori_y[i])
            train_ori_x.append(review)
        else:
            train_kw_x.append(review)
            train_kw_y.append(ori_y[i])
            train_ori_x.append(review)

    return train_ori_x, train_kw_x, train_kw_y


# 使用LDA中得到的关键词组成的字符串来表示句子
def represent_food_review_with_kw(ori_x, ori_y, keywords):
    train_ori_x = []
    train_kw_x = []
    train_kw_y = []
    for i in range(0, len(ori_x)):
        review = ori_x[i]
        contained_keywords = get_contained_keywords(review, keywords)
        if len(contained_keywords) != 0:
            train_kw_x.append(' '.join(contained_keywords))
            train_kw_y.append(ori_y[i])
            train_ori_x.append(review)

    return train_ori_x, train_kw_x, train_kw_y


# 获得训练、验证和测试集中的所有句子
def get_all_sentences(train_x, test_x, dev_x=None):
    sentences = []
    for sent in train_x:
        sentences.append(sent)
    if dev_x is not None:
        for sent in dev_x:
            sentences.append(sent)
    for sent in test_x:
        sentences.append(sent)
    return sentences


def split():
    samples = []
    with open("yelp_review_1/yelp_review8", 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(line)

    train = open("yelp_review_1/yelp_review8_train", 'w', encoding='utf-8')
    dev = open("yelp_review_1/yelp_review8_dev", 'w', encoding='utf-8')
    test = open("yelp_review_1/yelp_review8_test", 'w', encoding='utf-8')

    i = 0
    for sample in samples:
        if i < 45000:
            train.write(sample)
        elif 45000 <= i < 47500:
            test.write(sample)
        else:
            dev.write(sample)
        i += 1


def adjust_file_for_bert(file_path):
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '1-*-' in line:
                predict_y = line[-2]
                new_y = str(int(predict_y) + 1)
                new_line = line.replace('1-*-', '').replace(predict_y+'\n', '') + ' 1 ' + new_y + '\n'
                samples.append(new_line)
            elif '2-*-' in line:
                predict_y = line[-2]
                new_y = str(int(predict_y) + 1)
                new_line = line.replace('2-*-', '').replace(predict_y+'\n', '') + ' 2 ' + new_y + '\n'
                samples.append(new_line)
            elif '3-*-' in line:
                predict_y = line[-2]
                new_y = str(int(predict_y) + 1)
                new_line = line.replace('3-*-', '').replace(predict_y+'\n', '') + ' 3 ' + new_y + '\n'
                samples.append(new_line)
            elif '4-*-' in line:
                predict_y = line[-2]
                new_y = str(int(predict_y) + 1)
                new_line = line.replace('4-*-', '').replace(predict_y+'\n', '') + ' 4 ' + new_y + '\n'
                samples.append(new_line)
            elif '5-*-' in line:
                predict_y = line[-2]
                new_y = str(int(predict_y) + 1)
                new_line = line.replace('5-*-', '').replace(predict_y+'\n', '') + ' 5 ' + new_y + '\n'
                samples.append(new_line)

    with open(file_path, 'w', encoding='utf-8') as f:
        for line in samples:
            f.write(line)


# 从bert测试结果中找出判错的样本
def pick_out_wrong_test_samples():
    bert_labels = []
    with open('yelp_new_reviews/bert_test_result.tsv', 'r', encoding='utf-8') as f:
        for line in f:
            label = line[-2]
            bert_labels.append(label)

    wrong_samples = []
    with open('yelp_new_reviews/test', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(0, len(lines)):
            predict_y = bert_labels[i]
            line = lines[i]
            if predict_y + '\n' not in line:
                wrong_samples.append(line.replace('\n', '')+' '+predict_y+' \n')

    with open('yelp_new_reviews/bert_wrong_clf_samples', 'w', encoding='utf-8') as f:
        for sample in wrong_samples:
            f.write(sample)


# 从bert和我们模型中找出共同被判错的样本
def pick_out_same_wrong_samples():
    # 读取
    bert_wrong_samples = []
    our_wrong_samples = []
    same_wrong_samples = []
    with open('yelp_new_reviews/bert_wrong_clf_samples', 'r', encoding='utf-8') as f:
        for line in f:
            bert_wrong_samples.append(line)
    with open("yelp_new_reviews/our_wrong_clf_samples", 'r', encoding='utf-8') as f:
        for line in f:
            our_wrong_samples.append(line)

    # 提取相同判错例句
    for bert_line in bert_wrong_samples:
        for our_line in our_wrong_samples:
            bert_content = bert_line[:-6]
            our_content = our_line[:-5]
            if bert_content == our_content:
                same_line = bert_line.replace('\n', '')+our_line[-5:]
                same_wrong_samples.append(same_line)

    # 写入
    with open("yelp_new_reviews/same_wrong_clf_samples", 'w', encoding='utf-8') as f:
        for same_line in same_wrong_samples:
            f.write(same_line)


# 统计测试集中每一类判错数量（召回率）
def analyze_wrong_samples():
    cat1_wrong_samples = []
    cat2_wrong_samples = []
    cat3_wrong_samples = []
    cat4_wrong_samples = []
    cat5_wrong_samples = []
    with open('yelp_new_reviews/our_wrong_clf_samples', 'r', encoding='utf-8') as f:
        for line in f:
            ori_label = line[-4]
            if ori_label == '1':
                cat1_wrong_samples.append(line)
            elif ori_label == '2':
                cat2_wrong_samples.append(line)
            elif ori_label == '3':
                cat3_wrong_samples.append(line)
            elif ori_label == '4':
                cat4_wrong_samples.append(line)
            elif ori_label == '5':
                cat5_wrong_samples.append(line)

    cat1_samples = []
    cat2_samples = []
    cat3_samples = []
    cat4_samples = []
    cat5_samples = []
    with open('yelp_new_reviews/test', 'r', encoding='utf-8') as f:
        for line in f:
            label = line[-2]
            if label == '1':
                cat1_samples.append(line)
            elif label == '2':
                cat2_samples.append(line)
            elif label == '3':
                cat3_samples.append(line)
            elif label == '4':
                cat4_samples.append(line)
            elif label == '5':
                cat5_samples.append(line)

    with open('yelp_new_reviews/our_wrong_samples_analysis', 'w', encoding='utf-8') as f:
        for sample in cat1_wrong_samples:
            f.write(sample)
        f.write('-----------------------------\n\n')
        for sample in cat2_wrong_samples:
            f.write(sample)
        f.write('-----------------------------\n\n')
        for sample in cat3_wrong_samples:
            f.write(sample)
        f.write('-----------------------------\n\n')
        for sample in cat4_wrong_samples:
            f.write(sample)
        f.write('-----------------------------\n\n')
        for sample in cat5_wrong_samples:
            f.write(sample)
        f.write('-----------------------------\n\n')


# 重新拆分yelp训练集、测试集
def split_yelp_data():
    yelp_files = []
    yelp_files.append('yelp_review_1/yelp_review1')
    yelp_files.append('yelp_review_1/yelp_review2')
    yelp_files.append('yelp_review_1/yelp_review3')
    yelp_files.append('yelp_review_1/yelp_review4')
    yelp_files.append('yelp_review_1/yelp_review5')
    yelp_files.append('yelp_review_1/yelp_review6')

    samples = []
    cat1 = 0
    cat2 = 0
    cat3 = 0
    cat4 = 0
    cat5 = 0

    for file in yelp_files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                if '1\n' in line and cat1 < 20000:
                    samples.append(line)
                    cat1 += 1
                elif '2\n' in line and cat2 < 20000:
                    samples.append(line)
                    cat2 += 1
                elif '3\n' in line and cat3 < 20000:
                    samples.append(line)
                    cat3 += 1
                elif '4\n' in line and cat4 < 20000:
                    samples.append(line)
                    cat4 += 1
                elif '5\n' in line and cat5 < 20000:
                    samples.append(line)
                    cat5 += 1
            if cat1 == 20000 and cat2 == 20000 and cat3 == 20000 and cat4 == 20000 and cat5 == 20000:
                break
        if cat1 == 20000 and cat2 == 20000 and cat3 == 20000 and cat4 == 20000 and cat5 == 20000:
            break

    random.shuffle(samples)
    train = open('yelp_new_reviews/train_double', 'w', encoding='utf-8')
    test = open('yelp_new_reviews/test_double', 'w', encoding='utf-8')

    i = 0
    for sample in samples:
        if i < 95000:
            train.write(sample)
        elif i >= 95000:
            test.write(sample)
        i += 1


# 统计测试集中每一类被错误分为类别的数量
def analyze_wrong_cats():
    cat1_dict = {}
    cat2_dict = {}
    cat3_dict = {}
    cat4_dict = {}
    cat5_dict = {}
    with open('Amazon_reviews/bert_wrong_samples_analysis', 'r', encoding='utf-8') as f:
        for line in f:
            if '--------------\n' not in line and line != '\n':
                labels = line[-5:-2].split()
                true_label = labels[0]
                pred_label = labels[1]
                if true_label == '1':
                    if pred_label not in cat1_dict.keys():
                        cat1_dict[pred_label] = 1
                    else:
                        cat1_dict[pred_label] += 1
                elif true_label == '2':
                    if pred_label not in cat2_dict.keys():
                        cat2_dict[pred_label] = 1
                    else:
                        cat2_dict[pred_label] += 1
                elif true_label == '3':
                    if pred_label not in cat3_dict.keys():
                        cat3_dict[pred_label] = 1
                    else:
                        cat3_dict[pred_label] += 1
                elif true_label == '4':
                    if pred_label not in cat4_dict.keys():
                        cat4_dict[pred_label] = 1
                    else:
                        cat4_dict[pred_label] += 1
                elif true_label == '5':
                    if pred_label not in cat5_dict.keys():
                        cat5_dict[pred_label] = 1
                    else:
                        cat5_dict[pred_label] += 1
    print()


# 统计每一类样本数据的平均长度
def cal_cat_avg_length():
    cat1_length_sum = 0
    cat2_length_sum = 0
    cat3_length_sum = 0
    cat4_length_sum = 0
    cat5_length_sum = 0
    cat1_num = 0
    cat2_num = 0
    cat3_num = 0
    cat4_num = 0
    cat5_num = 0
    with open('Amazon_mobile_reviews/train', 'r', encoding='utf-8') as f:
        for line in f:
            if '1\n' in line:
                cat1_length_sum += len(line.replace('1\n', '').split())
                cat1_num += 1
            elif '2\n' in line:
                cat2_length_sum += len(line.replace('2\n', '').split())
                cat2_num += 1
            elif '3\n' in line:
                cat3_length_sum += len(line.replace('3\n', '').split())
                cat3_num += 1
            elif '4\n' in line:
                cat4_length_sum += len(line.replace('4\n', '').split())
                cat4_num += 1
            elif '5\n' in line:
                cat5_length_sum += len(line.replace('5\n', '').split())
                cat5_num += 1

    cat1_avg = cat1_length_sum / cat1_num
    cat2_avg = cat2_length_sum / cat2_num
    cat3_avg = cat3_length_sum / cat3_num
    cat4_avg = cat4_length_sum / cat4_num
    cat5_avg = cat5_length_sum / cat5_num
    print()


# 提取只被BERT判错或只被我们模型判错的样本
def extract_single_wrong_samples():
    all_wrong_samples = []  # 模型所有判错的样本
    with open('Amazon_reviews/bert_wrong_clf_samples', 'r', encoding='utf-8') as f:
        for line in f:
            all_wrong_samples.append(line)

    same_wrong_samples = []  # 两个模型都判错的样本
    with open('Amazon_reviews/same_wrong_clf_samples', 'r', encoding='utf-8') as f:
        for line in f:
            line = line[:-10]
            same_wrong_samples.append(line)

    single_wrong_samples = []  # 只有一个模型判错的样本
    for all_wrong_sample in all_wrong_samples:
        is_contained = False
        for same_wrong_sample in same_wrong_samples:
            if same_wrong_sample in all_wrong_sample:
                is_contained = True
                break
        if not is_contained:
            single_wrong_samples.append(all_wrong_sample)

    with open('Amazon_reviews/bert_single_wrong_samples', 'w', encoding='utf-8') as f:
        for line in single_wrong_samples:
            label = line[-5]
            if label == '1':
                f.write(line)
        for line in single_wrong_samples:
            label = line[-5]
            if label == '2':
                f.write(line)
        for line in single_wrong_samples:
            label = line[-5]
            if label == '3':
                f.write(line)
        for line in single_wrong_samples:
            label = line[-5]
            if label == '4':
                f.write(line)
        for line in single_wrong_samples:
            label = line[-5]
            if label == '5':
                f.write(line)
    print()


def draw_effect_of_kernel_size():
    x = ['2', '3', '4', '5', '6']
    y1 = [0.6292, 0.6276, 0.6236, 0.6200, 0.6140]
    y2 = [0.6132, 0.6132, 0.6116, 0.6064, 0.6036]

    plt.plot(x, y1, c='red', label='Yelp Reviews')
    plt.plot(x, y2, c='blue', label='Amazon Food Reviews')
    plt.xlabel('Kernel size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
