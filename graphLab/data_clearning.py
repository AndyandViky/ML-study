# -*- coding: utf-8 -*-

# /Users/yanglin/Desktop/yl/c++/arcsoft-arcface/face-api/script/articles.txt

import pandas as pd
import jieba

filePath = '/Users/yanglin/Desktop/yl/c++/arcsoft-arcface/face-api/script/articles.txt'
file = open(filePath)

datas = []
for line in file:
    test = line.split('@@@')
    test[1] = test[1].replace('"', '')
    if len(test[2]) <= 500:
        continue

    # 替换标点为空格，替换回车符等
    test[2] = test[2].replace('"', '').replace('\\n', ' ').replace('，', ' ')\
        .replace('、', ' ').replace('。', ' ').replace('\\t', '').replace(',', ' ')\
        .replace('\n', ' ')

    # 进行中文分词
    result = jieba.lcut(test[2], cut_all=True)
    result = ' '.join(result)

    datas.append([
        test[0],
        test[1],
        result
    ])

name = ['id', 'title', 'content']

pFrame = pd.DataFrame(columns=name, data=datas)

pFrame.to_csv('article.csv', encoding='utf-8')
