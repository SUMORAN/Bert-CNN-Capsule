# -*- encoding: utf-8 -*-


#import keras.backend.tensorflow_backend as KTF
import os
from keras.layers.normalization import BatchNormalization
#from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers.core import Dense, Dropout, Flatten, RepeatVector, K
from keras.layers import LSTM, Bidirectional, concatenate, Input, Embedding, Add, Multiply, Dot, GRU, Average
from keras.layers import Conv1D, GlobalMaxPool1D, GlobalAveragePooling1D, Lambda, Permute, Activation
from keras.activations import softmax, relu
# from attention import Attention
from gensim.models import word2vec
from data_preprocess import preprocess_yelp_clf_data, preprocess_imdb_clf_data, preprocess_dbpedia_data
from data_preprocess import preprocess_amazon_clf_data, preprocess_amazon_mobile_data, preprocess_MR_SST_data
from data_preprocess import sent_word_projection, augment_train_set
# from transformer import MyTransformerEncoder
import numpy as np
import random
import data_helper
import pandas as pd
import capsule

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''


train_file_path = "training_data/seg_sent_without_label"
word2vec_file_path = "output_result/up_down_stream.model"  # 词嵌入模型路径
vector_file_path = "output_result/up_down_stream.vector"  # 词嵌入向量路径
text_file_path = "training_data/seg_sent_with_label"  # 带标注的文本文件路路径

train_text_path = "train_test_data/train_text"  # 训练集句子
test_text_path = "train_test_data/test_text"  # 测试集句子
predicted_test_path = "train_test_data/predicted_test_text"  # 标签为被LSTM预测的句子


# 训练词向量模型
def train_word2vec():
    sentences = word2vec.Text8Corpus(train_file_path)
    model = word2vec.Word2Vec(sentences, sg=1, min_count=5, window=5, )

    model.save(word2vec_file_path)
    model.wv.save_word2vec_format(vector_file_path, binary=False)

    print("Finish training. Model saved!")
    sim_word = model.most_similar(["医院"], topn=20)
    for word in sim_word:
        for w in word:
            print(w)


# Yelp评论
def model_yelp_reviews(model_path, review_paths, max_len, dim):
    data_dict = preprocess_yelp_clf_data(model_path, review_paths, max_len, dim)

    # 从字典中提取词语数量和词嵌入矩阵
    vocab_size = data_dict['vocab_size']
    embedding_matrix = data_dict['embed']
    # 从字典中提取x
    train_x = data_dict['train_x']
    test_x = data_dict['test_x']

    # 从字典中提取y
    train_y = data_dict['train_y']
    test_y = data_dict['test_y']

    input = Input(shape=(max_len,), dtype='int32')  # 原始文本
    embed = Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=max_len, trainable=False)(input)
    embed_dp = Dropout(0.3)(embed)

    conv = Conv1D(filters=300, kernel_size=2, padding='same', activation='relu', strides=1)(embed_dp)
    conv_lstm = Bidirectional(LSTM(max_len,
                                   dropout=0.3,
                                   return_sequences=True,
                                   return_state=False))(conv)

    lstm = Bidirectional(LSTM(max_len,
                              dropout=0.3,
                              return_sequences=True,
                              return_state=False))(embed_dp)

    att = Multiply()([lstm, conv_lstm])
    pool = GlobalMaxPool1D()(att)

    predictions = Dense(5, activation='softmax')(pool)

    model = Model(inputs=input, outputs=predictions)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    print("Start training...")
    model.fit(train_x, train_y, batch_size=64, epochs=10, verbose=1, validation_data=[test_x, test_y])
    #model.save_weights('trained_model/yelp/CCG_w.h5')
    score = model.evaluate(test_x, test_y, verbose=2)
    print("Accuracy:" + str(score))


# Imdb模型
def model_imdb(model_path, train_path, test_path, max_len, kw_max_len, dim):
    data_dict = preprocess_imdb_clf_data(model_path, train_path, test_path, max_len, kw_max_len, dim)

    # 从字典中提取词语数量和词嵌入矩阵
    vocab_size = data_dict['vocab_size']
    embedding_matrix = data_dict['embed']

    # 从字典中提取x
    train_x = data_dict['train_x']
    test_x = data_dict['test_x']

    # 从字典中提取y
    train_y = data_dict['train_y']
    test_y = data_dict['test_y']

    input = Input(shape=(max_len,), dtype='int32')  # 原始文本
    embed = Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=max_len, trainable=False)(input)
    embed_dp = Dropout(0.3)(embed)

    conv = Conv1D(filters=300, kernel_size=3, padding='same', activation='relu', strides=1)(embed_dp)
    conv_lstm = Bidirectional(LSTM(max_len,
                                   dropout=0.3,
                                   return_sequences=True,
                                   return_state=False))(conv)

    lstm = Bidirectional(LSTM(max_len,
                              dropout=0.3,
                              return_sequences=True,
                              return_state=False))(embed_dp)

    att = Multiply()([lstm, conv_lstm])
    pool = GlobalMaxPool1D()(att)

    dense = Dense(2 * max_len, activation='relu')(pool)
    dp = Dropout(0.32)(dense)
    predictions = Dense(8, activation='softmax')(dp)

    model = Model(inputs=input, outputs=predictions)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    print("Start training...")
    model.fit(train_x, train_y, batch_size=128, epochs=10, verbose=1, validation_data=[test_x, test_y])
    model.save_weights('trained_model/imdb/ours_w.h5')
    score = model.evaluate(test_x, test_y, verbose=2)
    print("Accuracy:" + str(score))


# Amazon食品评论模型
def model_amazon_reviews(model_path, reviews_path, max_len, dim):
    data_dict = preprocess_amazon_clf_data(model_path, reviews_path, max_len, dim)

    # 从字典中提取词语数量和词嵌入矩阵
    vocab_size = data_dict['vocab_size']
    embedding_matrix = data_dict['embed']
    # 从字典中提取x
    train_x = data_dict['train_x']
    test_x = data_dict['test_x']

    # 从字典中提取y
    train_y = data_dict['train_y']
    test_y = data_dict['test_y']

    input = Input(shape=(max_len,), dtype='int32')  # 原始文本
    embed = Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=max_len, trainable=False)(input)
    embed_dp = Dropout(0.3)(embed)

    conv1 = Conv1D(filters=dim, kernel_size=3, padding='same', activation='relu', strides=1)(embed_dp)
    conv2 = Conv1D(filters=dim, kernel_size=3, padding='same', dilation_rate=2, activation='relu')(conv1)
    relu2 = Activation('relu')(conv2)

    conv3 = Conv1D(filters=dim, kernel_size=3, padding='same', dilation_rate=4, activation='relu')(relu2)
    relu3 = Activation('relu')(conv3)

    conv4 = Conv1D(filters=dim, kernel_size=3, padding='same', dilation_rate=8, activation='relu')(relu3)
    relu4 = Activation('relu')(conv4)

    conv5 = Conv1D(filters=dim, kernel_size=3, padding='same', dilation_rate=16, activation='relu')(relu4)
    relu5 = Activation('relu')(conv5)

    pool = GlobalMaxPool1D()(relu5)

    predictions = Dense(5, activation='softmax')(pool)
    model = Model(inputs=input, outputs=predictions)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    model.fit(train_x, train_y, batch_size=64, epochs=10, verbose=1, validation_data=[test_x, test_y])
    #model.save_weights('trained_model/amazon_food/CCG_w.h5')
    score = model.evaluate(test_x, test_y, verbose=2)
    print("Accuracy:" + str(score))


# Amazon电子设备评论模型
def model_mobile_reviews(model_path, review_paths, max_len, dim):
    data_dict = preprocess_amazon_mobile_data(model_path, review_paths, max_len, dim)

    # 从字典中提取词语数量和词嵌入矩阵
    vocab_size = data_dict['vocab_size']
    embedding_matrix = data_dict['embed']

    # 从字典中提取x
    train_x = data_dict['train_x']
    test_x = data_dict['test_x']

    # 从字典中提取y
    train_y = data_dict['train_y']
    test_y = data_dict['test_y']

    input = Input(shape=(max_len,), dtype='int32')  # 原始文本
    embed = Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=max_len, trainable=False)(input)
    embed_dp = Dropout(0.3)(embed)

    conv1 = Conv1D(filters=300, kernel_size=4, padding='valid', activation='relu', strides=4)(embed_dp)
    dp1 = Dropout(0.3)(conv1)
    mid_conv = Conv1D(filters=300, kernel_size=3, padding='same', activation='relu', strides=1)(dp1)
    dp2 = Dropout(0.3)(mid_conv)
    conv2 = Conv1D(filters=300, kernel_size=3, padding='same', dilation_rate=2, activation='relu')(dp2)
    dp3 = Dropout(0.3)(conv2)
    conv3 = Conv1D(filters=300, kernel_size=3, padding='same', dilation_rate=4, activation='relu')(dp3)
    dp4 = Dropout(0.3)(conv3)
    conv4 = Conv1D(filters=300, kernel_size=3, padding='same', dilation_rate=8, activation='relu')(dp4)

    flat = Flatten()(conv4)

    predictions = Dense(5, activation='softmax')(flat)

    model = Model(inputs=input, outputs=predictions)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    print("Start training...")
    model.fit(train_x, train_y, batch_size=32, epochs=40, verbose=1, validation_data=[test_x, test_y])
    #model.save_weights('trained_model/amazon_mobile/CCG_w.h5')
    score = model.evaluate(test_x, test_y, verbose=2)
    print("Accuracy:" + str(score))


# DBPedia主题分类
def model_dbpedia(model_path, review_paths, max_len, dim):
    data_dict = preprocess_dbpedia_data(model_path, review_paths, max_len, dim)

    # 从字典中提取词语数量和词嵌入矩阵
    vocab_size = data_dict['vocab_size']
    embedding_matrix = data_dict['embed']

    # 从字典中提取x
    train_x = data_dict['train_x']
    test_x = data_dict['test_x']

    # 从字典中提取y
    train_y = data_dict['train_y']
    test_y = data_dict['test_y']

    input = Input(shape=(max_len,), dtype='int32')  # 原始文本
    embed = Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=max_len, trainable=False)(input)
    embed_dp = Dropout(0.3)(embed)

    lstm = Bidirectional(LSTM(max_len, dropout=0.3))(embed_dp)

    dense = Dense(2 * max_len, activation='relu')(lstm)
    dp = Dropout(0.32)(dense)
    predictions = Dense(14, activation='softmax')(dp)

    model = Model(inputs=input, outputs=predictions)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    print("Start training...")
    model.fit(train_x, train_y, batch_size=128, epochs=10, verbose=1, validation_data=[test_x, test_y])
    model.save_weights('trained_model/dbpedia/lstm_w.h5')
    score = model.evaluate(test_x, test_y, verbose=2)
    print("Accuracy:" + str(score))


def model_yahoo(model_path, review_paths, max_len, dim):
    data_dict = preprocess_dbpedia_data(model_path, review_paths, max_len, dim)

    # 从字典中提取词语数量和词嵌入矩阵
    vocab_size = data_dict['vocab_size']
    embedding_matrix = data_dict['embed']

    # 从字典中提取x
    train_x = data_dict['train_x']
    test_x = data_dict['test_x']

    # 从字典中提取y
    train_y = data_dict['train_y']
    test_y = data_dict['test_y']

    input = Input(shape=(max_len,), dtype='int32')  # 原始文本
    embed = Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=max_len, trainable=False)(input)
    embed_dp = Dropout(0.3)(embed)

    lstm = Bidirectional(LSTM(max_len, dropout=0.3))(embed_dp)

    dense = Dense(2 * max_len, activation='relu')(lstm)
    dp = Dropout(0.32)(dense)
    predictions = Dense(14, activation='softmax')(dp)

    model = Model(inputs=input, outputs=predictions)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    print("Start training...")
    model.fit(train_x, train_y, batch_size=128, epochs=10, verbose=1, validation_data=[test_x, test_y])
    model.save_weights('trained_model/yahoo/lstm_w.h5')
    score = model.evaluate(test_x, test_y, verbose=2)
    print("Accuracy:" + str(score))


# 返回LSTM与自注意力结合的结果
def model_lstm_with_self_att(embed_dp, max_len):
    hidden_states = embed_dp
    hidden_states = Bidirectional(LSTM(max_len,
                                           dropout=0.3,
                                           return_sequences=True,
                                           return_state=False))(hidden_states)

    # Attention mechanism
    attention = Conv1D(filters=max_len, kernel_size=1, activation='tanh', padding='same', use_bias=True,
                       kernel_initializer='glorot_uniform', bias_initializer='zeros', name="attention_layer1")(hidden_states)
    attention = Conv1D(filters=max_len, kernel_size=1, activation='linear', padding='same',use_bias=True,
                       kernel_initializer='glorot_uniform', bias_initializer='zeros',
                       name="attention_layer2")(attention)
    attention = Lambda(lambda x: softmax(x, axis=1), name="attention_vector")(attention)

    # Apply attention weights
    weighted_sequence_embedding = Dot(axes=[1, 1], normalize=False, name="weighted_sequence_embedding")(
        [attention, hidden_states])

    # Add and normalize to obtain final sequence embedding
    sequence_embedding = Lambda(lambda x: K.l2_normalize(K.sum(x, axis=1)))(weighted_sequence_embedding)

    return sequence_embedding


# 测试Transformer
def test_transformer(model_path, review_paths, max_len, kw_max_len, dim):
    data_dict = preprocess_amazon_mobile_data(model_path, review_paths, max_len, kw_max_len)

    # 从字典中提取词语数量和词嵌入矩阵
    vocab_size = data_dict['vocab_size']
    embedding_matrix = data_dict['embed']

    # 从字典中提取x
    train_x = data_dict['train_x']
    test_x = data_dict['test_x']

    # 从字典中提取y
    train_y = data_dict['train_y']
    test_y = data_dict['test_y']

    # 提取LDA主题词
    #train_kw_x = data_dict['train_kw_x']
    #test_kw_x = data_dict['test_kw_x']

    print("Building model...")
    transformer = MyTransformerEncoder(vocab_size, max_len, kw_max_len, d_model=dim, word_embed_matrix=embedding_matrix)
    transformer.compile()
    print(transformer.model.summary())
    transformer.fit_model(train_x, train_y, )
    transformer.evaluate(test_x, test_y, )


#############################################################
# 利用BERT预训练模型
def model_yelp_via_bert(model_path, review_paths, max_len, dim):
    '''
    #data_dict = preprocess_yelp_clf_data(model_path, review_paths, max_len, dim)
    data_dict = sent_word_projection(model_path, review_paths, max_len, dim)

    # 从字典中提取词语数量和词嵌入矩阵
    vocab_size = data_dict['vocab_size']
    embedding_matrix = data_dict['embed']
    # 从字典中提取x
    train_x = data_dict['train_x']
    test_x = data_dict['test_x']

    # 提取投影
    train_proj = data_dict['train_proj']
    test_proj = data_dict['test_proj']

    # 从字典中提取y
    train_y = data_dict['train_y']
    test_y = data_dict['test_y']
    '''

    train_sent_embed = np.load('bert_embedding/train_embedding_sen.npy')
    test_sent_embed = np.load('bert_embedding/test_embedding_sen.npy')
    train_aug_sent_embed = np.load('bert_embedding/sim_same_label_pairs_sent_embed.npy')
    #train_word_embed = np.load('bert_embedding/train_embedding_word.npy')
    #test_word_embed = np.load('bert_embedding/test_embedding_word.npy')
    train_y = np.load('bert_embedding/train_label.npy')
    test_y = np.load('bert_embedding/test_label.npy')
    train_aug_y = np.load('bert_embedding/sim_pair_labels.npy')
    '''
    # 增强数据集与原数据集合并，并打乱顺序
    train = []
    for x, y in zip(train_sent_embed, train_y):
        train.append((x, y))
    for x, y in zip(train_aug_sent_embed, train_aug_y):
        train.append((x, y))
    random.shuffle(train)

    # 分离出训练数据的x和y
    train_x = []
    train_y = []
    for t in train:
        train_x.append(t[0])
        train_y.append(t[1])
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    '''
    print('Loaded embedding.')

    word_input = Input(shape=(dim, ), dtype='float32')

    predictions = Dense(5, activation='softmax')(word_input)

    model = Model(inputs=word_input, outputs=predictions)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    print("Start training...")
    model.fit(train_aug_sent_embed, train_aug_y, batch_size=128, epochs=5, verbose=1, validation_data=[test_sent_embed, test_y])
    score = model.evaluate(test_sent_embed, test_y, verbose=2)
    print("Accuracy:" + str(score))



def model_mobile_via_bert(max_len, dim):
    train_word_embed = np.load('bert_embedding_mobile/train_embedding_word.npy')
    test_word_embed = np.load('bert_embedding_mobile/test_embedding_word.npy')
    train_y = np.load('bert_embedding_mobile/train_label.npy')
    test_y = np.load('bert_embedding_mobile/test_label.npy')

    print('Loaded embedding.')

    word_input = Input(shape=(max_len, dim), dtype='float32')

    conv1 = Conv1D(filters=max_len * 2, kernel_size=3, padding='same', activation='relu', strides=1, use_bias=True)(
        word_input)
    conv2 = Conv1D(filters=max_len * 2, kernel_size=4, padding='same', activation='relu', strides=1, use_bias=True)(
        word_input)
    conv3 = Conv1D(filters=max_len * 2, kernel_size=5, padding='same', activation='relu', strides=1, use_bias=True)(
        word_input)
    pool1 = GlobalMaxPool1D()(conv1)
    pool2 = GlobalMaxPool1D()(conv2)
    pool3 = GlobalMaxPool1D()(conv3)
    pool_cat = concatenate([pool1, pool2, pool3], axis=-1)
    pool_dense = Dense(max_len * 2, activation='relu')(pool_cat)

    lstm = Bidirectional(LSTM(max_len, dropout=0.3))(word_input)
    lstm_dense = Dense(max_len * 2, activation='relu')(lstm)

    att = Multiply()([pool_dense, lstm_dense])
    predictions = Dense(5, activation='softmax')(att)

    model = Model(inputs=word_input, outputs=predictions)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    print("Start training...")
    model.fit(train_word_embed, train_y, batch_size=128, epochs=5, verbose=1, validation_data=[test_word_embed, test_y])
    score = model.evaluate(test_word_embed, test_y, verbose=2)
    print("Accuracy:" + str(score))


# 帮张靓跑的对比实验，MR数据集与SST数据集
def model_MR_SST_via_bert(max_len, dim):
    train_word_embed = np.load('../zhangliang/codes/bert-CNN-capsule/SST/SST5/train_embedding_word.npy')
    test_word_embed = np.load('../zhangliang/codes/bert-CNN-capsule/SST/SST5/test_embedding_word.npy')
    train_y = np.load('../zhangliang/codes/bert-CNN-capsule/SST/SST5/train_label.npy')
    test_y = np.load('../zhangliang/codes/bert-CNN-capsule/SST/SST5/test_label.npy')

    print('Loaded embedding.')

    input = Input(shape=(max_len, dim), dtype='float32')  # 原始文本

    conv1 = Conv1D(filters=max_len * 2, kernel_size=3, padding='same', activation='relu', strides=1, use_bias=True)(
        input)
    conv2 = Conv1D(filters=max_len * 2, kernel_size=4, padding='same', activation='relu', strides=1, use_bias=True)(
        input)
    conv3 = Conv1D(filters=max_len * 2, kernel_size=5, padding='same', activation='relu', strides=1, use_bias=True)(
        input)
    pool1 = GlobalMaxPool1D()(conv1)
    pool2 = GlobalMaxPool1D()(conv2)
    pool3 = GlobalMaxPool1D()(conv3)
    pool_cat = concatenate([pool1, pool2, pool3], axis=-1)
    pool_dense = Dense(max_len * 2, activation='relu')(pool_cat)

    lstm = Bidirectional(LSTM(max_len, dropout=0.3))(input)
    lstm_dense = Dense(max_len * 2, activation='relu')(lstm)

    att = Multiply()([pool_dense, lstm_dense])
    predictions = Dense(5, activation='softmax')(att)

    model = Model(inputs=input, outputs=predictions)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    print("Start training...")
    model.fit(train_word_embed, train_y, batch_size=128, epochs=10, verbose=1, validation_data=[test_word_embed, test_y])
    score = model.evaluate(test_word_embed, test_y, verbose=2)
    print("Accuracy:" + str(score))
    predicts = model.predict(test_word_embed, batch_size=128)
    predict_y = []
    for item in predicts:
        item = item.tolist()
        predict_y.append(item.index(max(item)))

    # print(predict_y)

    data_getter = data_helper.Dataprocessor_mr()
    test_examples, test_labels = data_getter.get_test_examples(data_dir="wrong_predicted_data")
    c = {
        "label": test_y,
        "predict_y": predict_y,
        "text": test_examples
    }

    df = pd.DataFrame(c)
    df_same = df[(df.label == df.predict_y)]
    df_different = df[(df.label != df.predict_y)]
    df_same.to_csv('right-rnn.csv', sep='\t', encoding='utf-8', index=False)
    df_different.to_csv('wrong-CRAN.csv', sep='\t', encoding='utf-8', index=False)
    print('save done!')


# 跑两个使用Glove词向量的对比实验
def model_MR_SST_via_glove(model_path, review_paths, max_len, dim):
    data_dict = preprocess_MR_SST_data(model_path, review_paths, max_len, dim)
    # data_dict = preprocess_yelp_clf_data(model_path, review_paths, max_len, dim)

    # 从字典中提取词语数量和词嵌入矩阵
    vocab_size = data_dict['vocab_size']
    embedding_matrix = data_dict['embed']

    # 从字典中提取x
    train_x = data_dict['train_x']
    test_x = data_dict['test_x']

    # 从字典中提取y
    train_y = data_dict['train_y']
    test_y = data_dict['test_y']

    input = Input(shape=(max_len,), dtype='int32')  # 原始文本
    embed = Embedding(vocab_size, dim, weights=[embedding_matrix], input_length=max_len, trainable=False)(input)
    # embed_dp = Dropout(0.3)(embed)

   #卷积层
    conv1 = Conv1D(filters=768, kernel_size=5, padding='same', activation='relu', strides=1)(embed)
    # conv2 = Conv1D(filters=768, kernel_size=4, padding='same', activation='relu', strides=1)(model_input)
    # conv3 = Conv1D(filters=768, kernel_size=4, padding='same', activation='relu', strides=1)(model_input)

    # conv = Average()([conv1,conv2,conv3])

    # 胶囊层
    caps0 = capsule.Capsule(
        num_capsule=60,
        dim_capsule=80,
        routings=5,
        share_weights=True,
        leaky=True)(conv1)
    # caps1 = capsule.Capsule(
    #     num_capsule=30,
    #     dim_capsule=40,
    #     routings=5,
    #     share_weights=True,
    #     leaky=args.leaky)(caps0)
    # flatten层
    flatten = Flatten()(caps0)
    # flatten = Dropout(args.dropout_rate)(flatten)
    # dense = Dense(3*NUM_CLASSES)(flatten)
    # softmax
    predictions = Dense(5, activation='softmax')(flatten)
    print(type(predictions))

    model = Model(inputs=input, outputs=predictions)
    # model = Model(inputs=[input1, input2], outputs=predictions)

    model.compile(loss='sparse_categorical_crossentropy', #binary_crossentropy
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())

    print("Start training...")
    model.fit(train_x, train_y, batch_size=128, epochs=10, verbose=1,
              validation_data=[test_x, test_y])
    score = model.evaluate(test_x, test_y, verbose=2)
    print("Accuracy:" + str(score))
    predicts = model.predict(test_x, batch_size=128)
    predict_y = []
    for item in predicts:
        item = item.tolist()
        predict_y.append(item.index(max(item)))

    # print(predict_y)

    data_getter = data_helper.Dataprocessor_yelp()
    test_examples, test_labels = data_getter.get_test_examples(data_dir="yelp_new_reviews")
    c = {
        "label": test_y,
        "predict_y": predict_y,
        "text": test_examples
    }

    df = pd.DataFrame(c)
    df_same = df[(df.label == df.predict_y)]
    df_different = df[(df.label != df.predict_y)]
    df_same.to_csv('right-cnn_caps.csv', sep='\t', encoding='utf-8', index=False)
    df_different.to_csv('yelp-cnn_caps.csv', sep='\t', encoding='utf-8', index=False)
    print('save done!')

# 预训练好的词向量模型
glove_300d_path = "glove.840B.300d.txt"
glove_100d_path = "yelp_glove/glove.6B.100d.txt"
ic_self_trained_model_path = "output_result/up_down_stream.vector"
fast_text_wiki_path = "fast_text_vec/wiki-news-300d-1M.vec"
fast_text_crawl_path = "fast_text_vec/crawl-300d-2M.vec"


# 停用词
stopwords_path = "yelp_glove/stop_words"
stopwords_path_1 = "yelp_glove/stop_words_1"


# 产业链数据
ic_path = "training_data/seg_sent_with_label2"
ic_paths = []
ic_paths.append(ic_path)

# IMDB影评训练、测试数据
imdb_train_path = 'Imdb_reviews/train_data'
imdb_test_path = 'Imdb_reviews/test_data'

# Amazon食品评论数据
amazon_reviews_path = 'Amazon_reviews/food_reviews.csv'
food_reviews = []
food_reviews.append('Amazon_reviews/train.txt')
food_reviews.append('Amazon_reviews/test.txt')

# Amazon电子设备评论数据
mobile_reviews = []
mobile_reviews.append('Amazon_mobile_reviews/train')
mobile_reviews.append('Amazon_mobile_reviews/test')
mobile_reviews.append('Amazon_mobile_reviews/dev')


# 文本蕴含数据
etm_path = "entailment_data/gov_data.txt"

# Yelp评论数据
yelp_paths = []
yelp_paths.append('yelp_new_reviews/train')
yelp_paths.append('yelp_new_reviews/test')

# DBPedia
dbpedia_paths = ['DBPedia/train', 'DBPedia/test']

# Yahoo
yahoo_paths = ['Yahoo/train', 'Yahoo/test']

# MR
mr_paths = ['MR/train', 'MR/test']
# SST-2
sst2_paths = ['SST2/train', 'SST2/test']
#SST-5
sst5_paths = ['SST5/train', 'SST5/test']
#yelp
yelp_paths = ['yelp_new_reviews/train', 'yelp_new_reviews/test']

if __name__ == '__main__':

    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #session = tf.Session(config=config)
    #KTF.set_session(session)

    #model_yelp_reviews(glove_300d_path, yelp_paths, max_len=120, dim=300)
    #model_imdb(glove_300d_path, imdb_train_path, imdb_test_path, max_len=230, kw_max_len=44, dim=300)
    #model_amazon_reviews(glove_300d_path, food_reviews, max_len=100, dim=300)
    #model_mobile_reviews(glove_300d_path, mobile_reviews, max_len=80, dim=300)
    #model_dbpedia(glove_300d_path, dbpedia_paths, max_len=60, dim=300)

    # BERT
    #model_yelp_via_bert(glove_300d_path, yelp_paths, 120, 768)
    #model_mobile_via_bert(120, 768)
    #model_MR_SST_via_bert(70, 768)

    model_MR_SST_via_glove(glove_300d_path, yelp_paths, 120, 300)


