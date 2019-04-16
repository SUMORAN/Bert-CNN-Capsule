# -*- encoding: utf-8 -*-

from keras.models import Model
from keras.layers.core import Dense
from keras.layers import LSTM, Bidirectional, Input, Multiply
from keras.layers import Conv1D, GlobalMaxPool1D
import data_helper


def model_yelp_reviews(max_len):
    train_x, train_y = data_helper.data_word_embedding('train')
    # train_y = np.array(train_y)
    # dev_x, dev_y = data_helper.data_embedding('dev')
    test_x, test_y = data_helper.data_word_embedding('test')
    # test_y = np.array(test_y)

    # 输入层
    input = Input(shape=(max_len, 768,), dtype='float32')

    #卷积层
    conv = Conv1D(filters=300, kernel_size=3, padding='same', activation='relu', strides=1)(input)
    conv_lstm = Bidirectional(LSTM(max_len,
                                   dropout=0.3,
                                   return_sequences=True,
                                   return_state=False))(conv)

    lstm = Bidirectional(LSTM(max_len,
                              dropout=0.3,
                              return_sequences=True,
                              return_state=False))(input)

    att = Multiply()([lstm, conv_lstm])
    pool = GlobalMaxPool1D()(att)

    predictions = Dense(5, activation='softmax')(pool)

    model = Model(inputs=input, outputs=predictions)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    print("Start training...")
    model.fit(train_x, train_y, batch_size=64, epochs=30, verbose=2, validation_data=[test_x, test_y])

    score = model.evaluate(test_x, test_y, verbose=2)
    print("Accuracy:" + str(score))   # 输出为loss和accuracy


model_yelp_reviews(120)
