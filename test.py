# -*- coding:utf-8 -*-


from keras.models import Model
import data_helper
import pandas as pd
from keras.layers.core import Dense
from keras.layers import Conv1D, Input, Flatten, GlobalMaxPool1D, GlobalAveragePooling1D
import numpy as np
import tensorflow as tf
import capsule
import os
from keras import callbacks
import model


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', default='yelp_result/news')
parser.add_argument('--batch_size', default=64, type=int)
args = parser.parse_args()

MAX_SEQUENCE_LEN = 120
NUM_CLASSES = 5

test_x, test_y = data_helper.data_word_embedding('test')

# 输入层
input = Input(shape=(MAX_SEQUENCE_LEN, 768,), dtype='float32')
'''
#卷积层
print('--------------卷积层---------')
# conv1 = Conv1D(filters=300, kernel_size=3, padding='same', activation='relu', strides=1)(input)
# conv2 = Conv1D(filters=300, kernel_size=3, padding='same', activation='relu', dilation_rate=2)(conv1)
# conv3 = Conv1D(filters=300, kernel_size=3, padding='same', activation='relu', dilation_rate=4)(conv2)
# conv4 = Conv1D(filters=300, kernel_size=3, padding='same', activation='relu', dilation_rate=8)(conv3)

print('--------------胶囊层---------')
caps = static_capsule.Capsule(
        num_capsule=50,
        dim_capsule=100,
        routings=5,
        share_weights=True)(input)

print('--------------flatten层---------')
flatten = Flatten()(caps)

# softmax
predictions = Dense(NUM_CLASSES, activation='softmax')(flatten)

model = Model(inputs=input, outputs=predictions)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())
'''
model = model.caps(input) # ,kernel_size=5
# load
model.load_weights(args.save_dir + '/weights-caps-12.h5')
predicts = model.predict(test_x, batch_size=args.batch_size)
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
df_same.to_csv('right-rnn.csv', sep='\t', encoding='utf-8', index=False)
df_different.to_csv('yelp-bert-caps.csv', sep='\t', encoding='utf-8', index=False)
print('save done!')
