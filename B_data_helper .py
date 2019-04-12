# -*- coding:utf-8 -*-

from bert_serving.client import BertClient
import os
import csv
import tokenization
import numpy as np
import json


embeddings = []
word_pieces = []
linenum = 1
bc = BertClient(ip='192.168.3.4')
with open('sim_pairs', "r", encoding='UTF-8') as f:
    for line in f:
        print('line {}',format(linenum))
        line = line.split(' ')
        text = tokenization.convert_to_unicode(' '.join(line[0:-1]))
        linenum += 1
        embedding, word_piece = bc.encode([text], show_tokens=True)
        embeddings.append(embedding)
        word_pieces.append(word_piece)

print('----Save train embeddings----')
np.save('embeddings', embeddings)
np.save('word_piece', word_piece)
