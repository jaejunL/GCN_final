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

with open("../../data/adjacency_matrix/ontology.json", "r") as st_json:
    ontology = json.load(st_json)

ontology_names = []
ontology_ids = []
ontology_childs = []
original_ontology_names = []

for i in range(len(ontology)):
    name = ontology[i]['name']
    name = name.replace(' and', '')
    name = name.replace(',', ' and')
    name = name.replace(' ', '_')
    ids = ontology[i]['id']
    childs = ontology[i]['child_ids']
    
    ontology_names.append(name)
    ontology_ids.append(ids)
    ontology_childs.append(childs)
    original_ontology_names.append(ontology[i]['name'])    

count = 0
parent_lists = []
for label in labels:
    temp = []
    label_id = ontology_ids[ontology_names.index(label)]
    for i in range(len(ontology_childs)):
        if label_id in ontology_childs[i]:
            child_idx = ontology_childs[i].index(label_id)
#             print('##label:', label, '##parent :', i, ontology_names[i], child_idx)
            temp.append(i)
            count += 1
    parent_lists.append(temp)
# print(count)    

ontology_adj1 = np.zeros([80, 80])
for i, parent in enumerate(parent_lists):
    for j, par in enumerate(parent):
        for k, temp_parent in enumerate(parent_lists):
            if par in temp_parent and i != k:
                ontology_adj1[i][k] += 1
                
data = {'adj': ontology_adj1}

with open('../../data/adjacency_matrix/ontology_adj1.pkl', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

count = 0
parent_lists = []
for ontology_id in ontology_ids:
    temp = []
    for i in range(len(ontology_childs)):
        if ontology_id in ontology_childs[i]:
            child_idx = ontology_childs[i].index(ontology_id)
#             print('##label:', label, '##parent :', i, ontology_names[i], child_idx)
            temp.append(i)
            count += 1
    parent_lists.append(temp)
# print(count)

ontology_adj2 = np.zeros([632, 632])
for i, parent in enumerate(parent_lists):
    for j, par in enumerate(parent):
        for k, temp_parent in enumerate(parent_lists):
            if par in temp_parent and i != k:
                ontology_adj2[i][k] += 1
                
data = {'adj': ontology_adj2}

with open('../../data/adjacency_matrix/ontology_adj2.pkl', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)