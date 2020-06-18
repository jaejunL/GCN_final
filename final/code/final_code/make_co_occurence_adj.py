import os
import json
import pickle
import numpy as np
import pandas as pd
from itertools import permutations, combinations

curated_csv_dir = '../../data/FSDKaggle2019.meta/train_curated_post_competition.csv'
noisy_csv_dir = '../../data/FSDKaggle2019.meta/train_noisy_post_competition.csv'

labels_csv = pd.read_csv("../../data/FSDKaggle2019.meta/labels.csv")
labels = labels_csv.columns[1:].to_list()

curated_csv = pd.read_csv(curated_csv_dir)
noisy_csv = pd.read_csv(noisy_csv_dir)

curated_num = np.zeros([80])
curated_adj = np.zeros([80, 80])

for i in range(len(curated_csv['labels'])):
    multi_labels = curated_csv['labels'][i].split(',')
    index_lists = []
    for multi_label in multi_labels:
        index_list = labels.index(multi_label)
        curated_num[index_list] += 1
        index_lists.append(index_list)
    subset_lists = list(permutations(index_lists, 2))
    for subset_list in subset_lists:
        curated_adj[subset_list[0]][subset_list[1]] += 1
#         print(subset_lists)

data = {'nums': curated_num,
        'adj': curated_adj
}

with open('../../data/adjacency_matrix/curated_adj.pkl', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


noisy_num = np.zeros([80])
noisy_adj = np.zeros([80, 80])

for i in range(len(noisy_csv['labels'])):
    multi_labels = noisy_csv['labels'][i].split(',')
    index_lists = []
    for multi_label in multi_labels:
        index_list = labels.index(multi_label)
        noisy_num[index_list] += 1
        index_lists.append(index_list)
    subset_lists = list(permutations(index_lists, 2))
    for subset_list in subset_lists:
        noisy_adj[subset_list[0]][subset_list[1]] += 1
#         print(subset_lists)

data = {'nums': curated_num + noisy_num,
        'adj': curated_adj + noisy_adj
}
with open('../../data/adjacency_matrix/curated_noisy_adj.pkl', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)    