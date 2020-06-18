import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import time, random, sys
import sklearn.metrics
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append('.')
from utils import AverageMeter, MelDataset_for_test, calculate_per_class_lwlrap
from models import ResNet, GCNResNet, GCN3ResNet

# set parameters
NUM_FOLD = 5
NUM_CLASS = 80
SEED = 42
NUM_EPOCH = 64*8
NUM_CYCLE = 64
BATCH_SIZE = 16
# LR = [1e-3, 1e-6]
CROP_LENGTH = 512

FOLD_LIST = [1, 2, 3, 4, 5]
EPOCH_LIST = [64, 128, 192, 256, 320, 384, 448, 512]
MODEL_DIR = "../../log/test"
MODEL_FOLDER = 'baseline'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FOLDER)
OUTPUT_DIR = os.path.join(MODEL_PATH, 'test')
cudnn.benchmark = True

starttime = time.time()

df_test = pd.read_csv("../../data/FSDKaggle2019.meta/test_post_competition.csv")
labels_csv = pd.read_csv("sample_submission.csv")

labels = labels_csv.columns[1:].to_list()

for label in labels:
    df_test[label] = df_test['labels'].apply(lambda x: label in x)
df_test['path'] = "../../data/dcase/mel128/test/" + df_test['fname']

if MODEL_FOLDER == 'baseline':
    MODEL = ResNet(NUM_CLASS)
elif MODEL_FOLDER == 'gcn1':
    adj_file = "../../data/adjacency_matrix/curated_adj.pkl"
    MODEL = GCNResNet(NUM_CLASS, t=0.4, adj_file=adj_file, in_channel=80)
elif MODEL_FOLDER == 'gcn2':
    adj_file = "../../data/adjacency_matrix/curated_noisy_adj.pkl"
    MODEL = GCNResNet(NUM_CLASS, t=0.4, adj_file=adj_file, in_channel=80)
elif MODEL_FOLDER == 'gcn3':
    adj_file = "../../data/adjacency_matrix/ontology_adj1.pkl"
    MODEL = GCN3ResNet(NUM_CLASS, adj_file=adj_file, in_channel=80)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(test_loader, model):
    bce_avr = AverageMeter()
    sigmoid = torch.nn.Sigmoid() #.cuda()
    criterion_bce = nn.BCEWithLogitsLoss() #.cuda()

    # switch to eval mode
    model.eval()

    inp = torch.zeros(80, 80, dtype=torch.float32)
    for i in range(len(inp)):
        inp[i][i] = 1
        
    # validate
    preds = np.zeros([0, NUM_CLASS], np.float32)
    y_true = np.zeros([0, NUM_CLASS], np.float32)
    for i, (input, target) in enumerate(test_loader):
        # get batches
        input = torch.autograd.Variable(input.to(device)) #.cuda())
        target = torch.autograd.Variable(target.to(device)) #.cuda())

        # compute output
        with torch.no_grad():
            if MODEL_FOLDER != 'baseline':
                output = model(input, inp.to(device))
            else:
                output = model(input)
            bce = criterion_bce(output, target)
            pred = sigmoid(output)
            pred = pred.data.cpu().numpy()

        # record log
        bce_avr.update(bce.data, input.size(0))
        preds = np.concatenate([preds, pred])
        y_true = np.concatenate([y_true, target.data.cpu().numpy()])

    # calc metric
    per_class_lwlrap, weight_per_class = calculate_per_class_lwlrap(y_true, preds)
    lwlrap = np.sum(per_class_lwlrap * weight_per_class)

    return bce_avr.avg.item(), lwlrap


log_columns = ['fold', 'test_bce', 'test_lwlrap', 'time']

test_log = pd.DataFrame(columns=log_columns)

for fold in FOLD_LIST:
    model = MODEL
    MODEL_NAME = 'weight_fold_' + str(fold) + '_best.pth'
    if os.path.isfile(os.path.join(MODEL_PATH, MODEL_NAME)):            
        model.load_state_dict(
            torch.load(os.path.join(MODEL_PATH, MODEL_NAME)))
    else:
        continue
    model.to(device)

    dataset_test = MelDataset_for_test(df_test['path'], df_test[labels].values,
                                crop=CROP_LENGTH)
    test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=1, pin_memory=True)

    test_bce, test_lwlrap = test(test_loader, model)

    # print log
    endtime = time.time() - starttime
    print("Fold :{} ".format(fold)
          + "Test CE: {:.4f} ".format(test_bce)
          + "Test LWLRAP: {:.4f} ".format(test_lwlrap)
          + "sec: {:.1f}".format(endtime)
          )

    # save log and weights
    test_log_epoch = pd.DataFrame(
        [[fold, test_bce, test_lwlrap, endtime]],
        columns=log_columns)
    test_log = pd.concat([test_log, test_log_epoch])
    test_log.to_csv("{}/test_log_best.csv".format(OUTPUT_DIR), index=False)
        