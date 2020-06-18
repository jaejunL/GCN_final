# GCN_final

This is repository of the final project of lecture Graph Convolutional Network (GCN) in Seoul National University (2020 spring).
It contains on-going process of Pytorch implementation of

Mt-Gcn For Multi-Label Audio Tagging With Noisy Labels (ICASSP 2020).

(https://ieeexplore.ieee.org/abstract/document/9053065)


# Usage
(1) First download and unzip FSDKaggle2019 dataset (https://zenodo.org/record/3612637#.XuvwsnUzbS8) into ```data/wav``` folder.

(2) Then run ```code/final_code/preprocess.py``` to generate mel-spectrogram input.

(as following procedure in https://github.com/OsciiArt/Freesound-Audio-Tagging-2019)

(3) Generate 3 types of adjacency matrix using ```make_co_occurence_adj.py``` and ```make_ontology_adj.py```.

(4) run ```baseline.py```, ```gcn1.py```, ```gcn2.py``` and ```gcn3.py```

(5) run ```inference.py``` for evaluation.
