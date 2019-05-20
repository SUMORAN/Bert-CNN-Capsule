# -*- encoding: utf-8 -*-

from util import load_word_ebd, split_train_test_for_yelp, split_train_test_for_text_etm, rm_useless_tokens, get_all_sentences
from util import read_dbpedia_data, read_imdb_data, represent_food_review_with_kw
from util import split_train_test_for_amazon, split_yelp_train_data_to_cats, represent_review_with_kw, split_imdb_train_data_to_cats
from util import split_train_test_for_mobile, split_train_test_for_db_amazon, split_train_test_for_zhang
# from eda import eda
# from seq_seg import seg_sequences
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
import random

neg_words_path = 'yelp_glove/negative_words'


# 将yelp数据集训练、验证、测试集以及词嵌入转化为网络所需的输入形式
def preprocess_yelp_clf_data(model_path, review_paths, max_len, dim):
    # 读取预训练词向量
    word_embed_dict = load_word_ebd(model_path, loading_num=1000000)
    # 划分训练、验证、测试集，该步中只移除了无用的标点，注意此步骤中并未去除停用词
    train_x, train_y, test_x, test_y = split_train_test_for_yelp(review_paths)

    # 将测试、验证、训练集的所有句子合并，用于下一步统计词语数量
    sentences = get_all_sentences(train_x, test_x)

    # 统计词语总数
    print("Begin to transform text to int nd-array...")
    t = Tokenizer()
    t.fit_on_texts(sentences)
    vocab_size = len(t.word_index) + 1

    # 将原始句子文本转化为整数token表示形式，此处若是与LDA一同输入，则需要使用 train_ori_x 而非 train_x
    encoded_train_docs = t.texts_to_sequences(train_x)
    encoded_test_docs = t.texts_to_sequences(test_x)

    # 将上述原始句子的整数token表示形式转化为 nd-array
    padded_train_docs = pad_sequences(encoded_train_docs, maxlen=max_len, padding='post')
    padded_test_docs = pad_sequences(encoded_test_docs, maxlen=max_len, padding='post')

    # 将出现在样本中的词语以及对应词向量加入字典
    embedding_matrix = np.zeros((vocab_size, dim))
    count = 0
    for word, i in t.word_index.items():
        try:
            vec = word_embed_dict[word]
            embedding_matrix[i] = vec
            count += 1
            print("Constructing embedding matrix.Finish:" + str(count))
        except KeyError:
            continue

    dic = {}
    dic['vocab_size'] = vocab_size
    dic['train_x'] = padded_train_docs
    dic['train_y'] = train_y
    dic['test_x'] = padded_test_docs
    dic['test_y'] = test_y
    dic['embed'] = embedding_matrix

    return dic


# 将IMDB影评数据划分为训练、验证、测试集以及词嵌入转化为网络所需的输入形式
def preprocess_imdb_clf_data(model_path, imdb_train_path, imdb_test_path, max_len, kw_max_len, dim):
    # 读取预训练词向量
    word_embed_dict = load_word_ebd(model_path, loading_num=1000000)

    train_x, train_y = read_imdb_data(imdb_train_path)
    test_x, test_y = read_imdb_data(imdb_test_path)

    train_x = train_x[:10000]
    train_y = train_y[:10000]

    test_x = test_x[:2500]
    test_y = test_y[:2500]

    sentences = get_all_sentences(train_x, test_x)

    print("Begin to transform text into int nd-array...")
    t = Tokenizer()
    t.fit_on_texts(sentences)
    vocab_size = len(t.word_index) + 1

    # 将原始句子文本转化为整数token表示形式，此处若是与LDA一同输入，则需要使用 train_ori_x 而非 train_x
    encoded_train_docs = t.texts_to_sequences(train_x)
    encoded_test_docs = t.texts_to_sequences(test_x)

    # 将上述原始句子的整数token表示形式转化为 nd-array
    padded_train_docs = pad_sequences(encoded_train_docs, maxlen=max_len, padding='post')
    padded_test_docs = pad_sequences(encoded_test_docs, maxlen=max_len, padding='post')

    # 将出现在样本中的词语以及对应词向量加入字典
    embedding_matrix = np.zeros((vocab_size, dim))
    count = 0
    for word, i in t.word_index.items():
        try:
            vec = word_embed_dict[word]
            embedding_matrix[i] = vec
            count += 1
            print("Constructing embedding matrix. Finish:" + str(count))
        except KeyError:
            continue

    dic = {}
    dic['vocab_size'] = vocab_size
    dic['train_x'] = padded_train_docs
    dic['train_y'] = train_y
    dic['test_x'] = padded_test_docs
    dic['test_y'] = test_y
    dic['embed'] = embedding_matrix

    return dic


# 将Amazon食品评论数据划分为训练、验证、测试集，并转化为网络所需的输入形式
def preprocess_amazon_clf_data(model_path, reviews_path, max_len, dim):
    # 读取预训练词向量
    word_embed_dict = load_word_ebd(model_path, loading_num=1000000)
    # 划分训练、验证、测试集，该步中只移除了无用的标点，注意此步骤中并未去除停用词
    train_x, train_y, test_x, test_y = split_train_test_for_db_amazon(reviews_path)

    # 将测试、验证、训练集的所有句子合并，用于下一步统计词语数量
    sentences = get_all_sentences(train_x, test_x)
    seg_sequences(word_embed_dict, sentences)

    # 统计词语总数
    print("Begin to transform text to int nd-array...")
    t = Tokenizer()
    t.fit_on_texts(sentences)
    vocab_size = len(t.word_index) + 1

    # 将原始句子文本转化为整数token表示形式，此处若是与LDA一同输入，则需要使用 train_ori_x 而非 train_x
    encoded_train_docs = t.texts_to_sequences(train_x)
    encoded_test_docs = t.texts_to_sequences(test_x)

    # 将上述原始句子的整数token表示形式转化为 nd-array
    padded_train_docs = pad_sequences(encoded_train_docs, maxlen=max_len, padding='post')
    padded_test_docs = pad_sequences(encoded_test_docs, maxlen=max_len, padding='post')

    # 将出现在样本中的词语以及对应词向量加入字典
    embedding_matrix = np.zeros((vocab_size, dim))
    count = 0
    for word, i in t.word_index.items():
        try:
            vec = word_embed_dict[word]
            embedding_matrix[i] = vec
            count += 1
            print("Constructing embedding matrix.Finish:" + str(count))
        except KeyError:
            continue

    dic = {}
    dic['vocab_size'] = vocab_size

    dic['train_x'] = padded_train_docs
    dic['train_y'] = train_y

    dic['test_x'] = padded_test_docs
    dic['test_y'] = test_y

    dic['embed'] = embedding_matrix

    return dic


# Amazon电子设备评论
def preprocess_amazon_mobile_data(model_path, review_paths, max_len, dim):
    # 读取预训练词向量
    eg_x, eg_y = load_eg()
    word_embed_dict = load_word_ebd(model_path, loading_num=1000000)
    # 划分训练、验证、测试集，该步中只移除了无用的标点，注意此步骤中并未去除停用词
    train_x, train_y, test_x, test_y, dev_x, dev_y = split_train_test_for_mobile(review_paths)
    # 后续会将训练集当做参数传入处理而改变其原始内容，提前将内容赋值

    # 将测试、验证、训练集的所有句子合并，用于下一步统计词语数量
    train_x = seg_sequences(word_embed_dict, train_x)
    test_x = seg_sequences(word_embed_dict, test_x)
    sentences = get_all_sentences(train_x, test_x, dev_x)

    # 统计词语总数
    print("Begin to transform text to int nd-array...")
    t = Tokenizer()
    t.fit_on_texts(sentences)
    vocab_size = len(t.word_index) + 1

    # 将原始句子文本转化为整数token表示形式，此处若是与LDA一同输入，则需要使用 train_ori_x 而非 train_x
    encoded_train_docs = t.texts_to_sequences(train_x)
    encoded_test_docs = t.texts_to_sequences(test_x)
    encoded_eg_docs = t.texts_to_sequences(eg_x)

    # 将上述原始句子的整数token表示形式转化为 nd-array
    padded_train_docs = pad_sequences(encoded_train_docs, maxlen=max_len, padding='post')
    padded_test_docs = pad_sequences(encoded_test_docs, maxlen=max_len, padding='post')
    padded_eg_docs = pad_sequences(encoded_eg_docs, maxlen=max_len, padding='post')

    # 将出现在样本中的词语以及对应词向量加入字典
    embedding_matrix = np.zeros((vocab_size, dim))
    count = 0
    for word, i in t.word_index.items():
        try:
            vec = word_embed_dict[word]
            embedding_matrix[i] = vec
            count += 1
            print("Constructing embedding matrix.Finish:" + str(count))
        except KeyError:
            continue

    dic = {}
    dic['vocab_size'] = vocab_size

    dic['train_x'] = padded_train_docs
    dic['train_y'] = train_y
    dic['test_x'] = padded_test_docs
    dic['test_y'] = test_y
    dic['eg_x'] = padded_eg_docs
    dic['eg_y'] = eg_y

    dic['embed'] = embedding_matrix

    return dic



def preprocess_MR_SST_data(model_path, review_paths, max_len, dim):

    word_embed_dict = load_word_ebd(model_path, loading_num=1000000)
    # 划分训练、验证、测试集，该步中只移除了无用的标点，注意此步骤中并未去除停用词
    train_x, train_y, test_x, test_y = split_train_test_for_zhang(review_paths)
    # 后续会将训练集当做参数传入处理而改变其原始内容，提前将内容赋值

    # 将测试、验证、训练集的所有句子合并，用于下一步统计词语数量
    # train_x = seg_sequences(word_embed_dict, train_x)
    # test_x = seg_sequences(word_embed_dict, test_x)
    sentences = get_all_sentences(train_x, test_x)

    # 统计词语总数
    print("Begin to transform text to int nd-array...")
    t = Tokenizer()
    t.fit_on_texts(sentences)
    vocab_size = len(t.word_index) + 1

    # 将原始句子文本转化为整数token表示形式，此处若是与LDA一同输入，则需要使用 train_ori_x 而非 train_x
    encoded_train_docs = t.texts_to_sequences(train_x)
    encoded_test_docs = t.texts_to_sequences(test_x)

    # 将上述原始句子的整数token表示形式转化为 nd-array
    padded_train_docs = pad_sequences(encoded_train_docs, maxlen=max_len, padding='post')
    padded_test_docs = pad_sequences(encoded_test_docs, maxlen=max_len, padding='post')

    # 将出现在样本中的词语以及对应词向量加入字典
    embedding_matrix = np.zeros((vocab_size, dim))
    count = 0
    for word, i in t.word_index.items():
        try:
            vec = word_embed_dict[word]
            embedding_matrix[i] = vec
            count += 1
            print("Constructing embedding matrix.Finish:" + str(count))
        except KeyError:
            continue

    dic = {}
    dic['vocab_size'] = vocab_size
    dic['train_x'] = padded_train_docs
    dic['train_y'] = train_y
    dic['test_x'] = padded_test_docs
    dic['test_y'] = test_y
    dic['embed'] = embedding_matrix

    return dic


# DBPedia主题
def preprocess_dbpedia_data(model_path, review_paths, max_len, dim):
    # 读取预训练词向量
    word_embed_dict = load_word_ebd(model_path, loading_num=1000000)

    train_x, train_y = read_dbpedia_data(review_paths[0])
    test_x, test_y = read_dbpedia_data(review_paths[1])

    sentences = get_all_sentences(train_x, test_x)

    print("Begin to transform text into int nd-array...")
    t = Tokenizer()
    t.fit_on_texts(sentences)
    vocab_size = len(t.word_index) + 1

    # 将原始句子文本转化为整数token表示形式，此处若是与LDA一同输入，则需要使用 train_ori_x 而非 train_x
    encoded_train_docs = t.texts_to_sequences(train_x)
    encoded_test_docs = t.texts_to_sequences(test_x)

    # 将上述原始句子的整数token表示形式转化为 nd-array
    padded_train_docs = pad_sequences(encoded_train_docs, maxlen=max_len, padding='post')
    padded_test_docs = pad_sequences(encoded_test_docs, maxlen=max_len, padding='post')

    # 将出现在样本中的词语以及对应词向量加入字典
    embedding_matrix = np.zeros((vocab_size, dim))
    count = 0
    for word, i in t.word_index.items():
        try:
            vec = word_embed_dict[word]
            embedding_matrix[i] = vec
            count += 1
            print("Constructing embedding matrix. Finish:" + str(count))
        except KeyError:
            continue

    dic = {}
    dic['vocab_size'] = vocab_size
    dic['train_x'] = padded_train_docs
    dic['train_y'] = train_y
    dic['test_x'] = padded_test_docs
    dic['test_y'] = test_y
    dic['embed'] = embedding_matrix

    return dic


# 将IMDB训练和测试数据分别读进两个文件中
def read_imdb_to_one_file():
    test_neg_path = 'Imdb/aclImdb/test/neg'
    test_pos_path = 'Imdb/aclImdb/test/pos'
    train_neg_path = 'Imdb/aclImdb/train/neg'
    train_pos_path = 'Imdb/aclImdb/train/pos'

    test_target_path = 'Imdb/test_data'
    train_target_path = "Imdb/train_data"

    test_neg_files = os.listdir(test_neg_path)
    test_pos_files = os.listdir(test_pos_path)
    train_neg_files = os.listdir(train_neg_path)
    train_pos_files = os.listdir(train_pos_path)

    test_reviews = []
    train_reviews = []
    i = 0
    for file in test_neg_files:
        with open(test_neg_path+'/'+file, 'r', encoding='utf-8') as f:
            rating = file.split('_')[1].replace('.txt', '')
            test_reviews.append(f.readlines()[0]+" "+rating+'\n')
        print(i)
        i += 1
    for file in test_pos_files:
        with open(test_pos_path+'/'+file, 'r', encoding='utf-8') as f:
            rating = file.split('_')[1].replace('.txt', '')
            test_reviews.append(f.readlines()[0]+" "+rating+'\n')
        print(i)
        i += 1

    for file in train_neg_files:
        with open(train_neg_path+'/'+file, 'r', encoding='utf-8') as f:
            rating = file.split('_')[1].replace('.txt', '')
            train_reviews.append(f.readlines()[0]+' '+rating+'\n')
        print(i)
        i += 1
    for file in train_pos_files:
        with open(train_pos_path+'/'+file, 'r', encoding='utf-8') as f:
            rating = file.split('_')[1].replace('.txt', '')
            train_reviews.append(f.readlines()[0]+' '+rating+'\n')
        print(i)
        i += 1

    random.shuffle(test_reviews)
    random.shuffle(train_reviews)

    with open(test_target_path, 'w', encoding='utf-8') as f:
        for review in test_reviews:
            f.write(review)
    with open(train_target_path, 'w', encoding='utf-8') as f:
        for review in train_reviews:
            f.write(review)

    print()


def load_eg():
    list = []
    labels = []
    with open('debug_data/amazon_mobile_data', 'r', encoding='utf-8') as f:
        for line in f:
            if ' 1\n' in line:
                labels.append(1)
            elif ' 2\n' in line:
                labels.append(2)
            elif ' 3\n' in line:
                labels.append(3)
            elif ' 4\n' in line:
                labels.append(4)
            elif ' 5\n' in line:
                labels.append(5)
            new_sent = rm_useless_tokens(line[:-2].lower())
            list.append(new_sent)

    return list, labels


########################################################################################
# 读取bert生成的词向量和句向量文件
# 将句子中的词向量和句向量做映射
def sent_word_projection(bert_word_embed, bert_sent_embed):
    sent_word_projs = []

    # 计算投影
    for i in range(0, len(bert_word_embed)):
        sent_word_embed = bert_word_embed[i]
        sent_embed = bert_sent_embed[i]

        sim = project(sent_word_embed, sent_embed)
        sent_word_projs.append(sim)
        print('Finished projecting sentence:'+str(i))

    return np.array(sent_word_projs)


# 对于一个句子，让句子中的每个词与句向量进行映射
def project(sent_word_embed, sent_embed):
    sims = []
    for word_embed in sent_word_embed:
        sim = cosine_similarity(word_embed, sent_embed)
        sims.append(sim)

    return np.array(sims)


# 计算余弦相似度
def cosine_similarity(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return round(dot_product / ((normB ** 0.5)*(normA ** 0.5)), 2)


###################################################################################
# 根据bert生成的句向量来计算句子之间的相似性，来扩充训练集
def augment_train_set(train_sent_embed, train_labels, train_sents):
    label1_set = []
    label2_set = []
    label3_set = []
    label4_set = []
    label5_set = []
    i = 0
    for sent_embed, y in zip(train_sent_embed, train_labels):
        if y == 0:
            label1_set.append(i)
        elif y == 1:
            label2_set.append(i)
        elif y == 2:
            label3_set.append(i)
        elif y == 3:
            label4_set.append(i)
        elif y == 4:
            label5_set.append(i)
        i += 1

    sim_pair_set1 = cal_sim_for_single_class(label1_set, label4_set+label5_set, train_sent_embed, train_sents)
    sim_pair_set2 = cal_sim_for_single_class(label2_set, label4_set+label5_set, train_sent_embed, train_sents)
    sim_pair_set3 = cal_sim_for_single_class(label3_set, label1_set+label5_set, train_sent_embed, train_sents)
    sim_pair_set4 = cal_sim_for_single_class(label4_set, label1_set+label2_set, train_sent_embed, train_sents)
    sim_pair_set5 = cal_sim_for_single_class(label5_set, label1_set+label2_set, train_sent_embed, train_sents)

    total_res = []
    for _ in sim_pair_set1:
        total_res.append(_)
    for _ in sim_pair_set2:
        total_res.append(_)
    for _ in sim_pair_set3:
        total_res.append(_)
    for _ in sim_pair_set4:
        total_res.append(_)
    for _ in sim_pair_set5:
        total_res.append(_)

    with open('bert_embedding/sim_pairs.txt', 'w', encoding='utf-8') as f:
        for _ in total_res:
            f.write(str(_[0])+' '+str(_[1])+'\n')


# 对于同一类别的句子，计算相似性，将最相似的一对句子的向量拼接
def cal_sim_for_single_class(class_set, comp_set, sent_embed, sents):
    random.shuffle(comp_set)
    res = []
    num = 0
    for i in class_set:
        embed1 = sent_embed[i]
        max_cos = 0
        max_sim_j = 0
        for j in comp_set:
            embed2 = sent_embed[j]
            cur_cos = cosine_similarity(embed1, embed2)
            if cur_cos > max_cos:
                max_cos = cur_cos
                max_sim_j = j
                if max_cos > 0.80:
                    break
            elif comp_set.index(j) > 100:
                break
            print('Finished processing:'+str(class_set.index(i))+' '+str(comp_set.index(j)))
        res.append((i, max_sim_j))
        print('Finished processing:'+str(num))
        num += 1
    return res


###################################################################################
# 根据词向量来计算句子的n-gram向量
def cal_n_gram_embedding(word_embed):
    n_gram_embedding = []
    j = 0
    for e in word_embed:
        sent_n_gram = []
        for i in range(0, len(e)-2):
            avg_e = (e[i] + e[i+1] + e[i+2])/3
            sent_n_gram.append(avg_e)
        print('Finished generating sent n-gram:'+str(j))
        j += 1
        n_gram_embedding.append(sent_n_gram)
    np.save('bert_embedding/test_embedding_n_gram.npy', n_gram_embedding)


# 计算n-gram向量与句向量的投影
def cal_n_gram_sent_sim(bert_n_gram_embed, bert_sent_embed):
    sent_n_gram_projs = []

    # 计算投影
    for i in range(0, len(bert_n_gram_embed)):
        sent_n_gram_embed = bert_n_gram_embed[i]
        sent_embed = bert_sent_embed[i]

        sim = project(sent_n_gram_embed, sent_embed)
        sent_n_gram_projs.append(sim)
        print('Finished projecting sentence:' + str(i))

    np.save('bert_embedding/train_n_gram_proj.npy', np.array(sent_n_gram_projs))



##################################################################################
# 根据bert生成的句向量比较句子相似度
def cal_sentence_sim(sent_embed):
    sim_pairs = []
    for i in range(len(sent_embed)):
        embed = sent_embed[i]
        sim = -1
        sim_id = 0
        for j in range(len(sent_embed)):
            embed2 = sent_embed[j]
            if (embed == embed2).all():
                continue
            cur_sim = cosine_similarity(embed, embed2)
            if sim < cur_sim:
                sim = cur_sim
                sim_id = j
                if cur_sim > 0.8:
                    break
            print('Finished sentence:'+str(i)+' '+str(j))
        sim_pairs.append((i, sim_id))
        print('Finished sentence:'+str(i))

    with open('bert_embedding/sim_pairs.txt', 'w', encoding='utf-8') as f:
        for pair in sim_pairs:
            f.write(str(pair[0])+' '+str(pair[1]))
            f.write('\n')


def merge_sim_sentences(sents, sim_indexes):
    f = open('bert_embedding/sim_same_label_pairs', 'w', encoding='utf-8')
    f2 = open('bert_embedding/sim_diff_label_pairs', 'w', encoding='utf-8')
    i = 0
    for pair in sim_indexes:
        sent1 = sents[int(pair[0])]
        sent2 = sents[int(pair[1])]
        label1, label2 = sent1[-1:], sent2[-1:]
        if label1 == label2:
            #text1 = sent1[:-1]
            #text2 = sent2[:-1]
            f.write(str(sent1)+' | | | '+str(sent2)+'\n')
        else:
            f2.write(str(sent1) + ' | | | ' + str(sent2) + '\n')

        print('Finish pair:'+str(i))
        i += 1
    print()


# 查看一下相似句子对在语义上是否相似
def look_sim_pairs(pairs, sents):
    with open('bert_embedding/sim_diff_label_pairs', 'w', encoding='utf-8') as f:
        for pair in pairs:
            sent1 = sents[int(pair[0])].strip()
            sent2 = sents[int(pair[1])].strip()
            f.write(sent1+' | | | '+sent2+'\n')


if __name__ == '__main__':

    #train_word_embed = np.load('bert_embedding/train_embedding_word.npy')
    #test_word_embed = np.load('bert_embedding/test_embedding_word.npy')

    #train_n_gram_embed = np.load('bert_embedding/train_embedding_n_gram.npy')
    #test_n_gram_embed = np.load('bert_embedding/test_embedding_n_gram.npy')

    train_sent_embed = np.load('bert_embedding/train_embedding_sen.npy')
    #test_sent_embed = np.load('bert_embedding/test_embedding_sen.npy')

    train_labels = np.load('bert_embedding/train_label.npy')

    glove_300d_path = "yelp_glove/glove.840B.300d.txt"
    data_paths = []
    data_paths.append('MR/train')
    data_paths.append('MR/test')

    preprocess_MR_SST_data(glove_300d_path, data_paths, 300, 90)
