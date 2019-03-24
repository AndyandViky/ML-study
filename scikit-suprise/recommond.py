# -*- coding: utf-8 -*-
import io
from surprise import KNNBaseline
from surprise import Dataset, Reader

def read_iter_names():
    # u.item格式：编号|电影名字|评分|url
    # 获取电影名到id和id到电影名的映射
    item_file = 'your path+/ml-100k/u.item'
    rid_2_name = {}
    name_2_rid = {}
    with io.open(item_file,'r',encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_2_name[line[0]]=line[1]
            name_2_rid[line[1]]=line[0]
    return rid_2_name,name_2_rid

# u.data数据格式为 user item rating timestamp；
reader = Reader(line_format='user item rating timestamp', sep='\t')
file_path = 'your path + /ml-100k'
data = Dataset.load_from_file(file_path=file_path+'/u.data',reader=reader)
train_set = data.build_full_trainset()
sim_options = {'name': 'pearson_baseline', 'user_based': False}
algo = KNNBaseline(sim_options=sim_options)
algo.train(train_set)

# 获取id对应的电影名列表，由于中途涉及一个id转换，所以要双向
rid_2_name,name_2_rid = read_iter_names()
# print(rid_2_name['1'])
# print(name_2_rid['Toy Story (1995)'])

# raw-id映射到内部id
toy_story_raw_id = name_2_rid['Toy Story (1995)']
toy_story_inner_id = algo.trainset.to_inner_iid(toy_story_raw_id)

# 获取toy story对应的内部id 并由此取得其对应的k个近邻 k个近邻对应的也是内部id
toy_story_neighbors  = algo.get_neighbors(toy_story_inner_id,k = 10)

# 近邻内部id转换为对应的名字
toy_story_neighbors = (algo.trainset.to_raw_iid(inner_id)
    for inner_id in toy_story_neighbors)
toy_story_neighbors = (rid_2_name[rid] for rid in toy_story_neighbors)

print('基于皮尔逊相似计算得到与toy story相近的十个电影为：\n')
for moives in toy_story_neighbors:
    print(moives)