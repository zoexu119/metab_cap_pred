from sklearn.metrics import auc,  precision_recall_curve, roc_auc_score, mean_absolute_error, mean_squared_error
import numpy as np
import csv
import codecs
import re
import os


def prc_auc(targets, preds):
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)


def smile_label_reader(file_name):
    inputs = []
    labels = []   
    with codecs.open(file_name, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = row['smiles']
            label = int(row['labels'])
            inputs.append(smiles)
            labels.append(label)
        return inputs, np.array(labels)

def graph_preds_reader(file_name):
    preds = []
    with codecs.open(file_name, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred = float(row['preds'])
            preds.append(pred)
        return np.array(preds)


seq_dir = '../seq/results/atcc15579_oversample_10/'
graph_dir = '../graph/results/atcc15579_oversample_10/'

seq_prcs = []
seq_rocs = []
graph_prcs = []
graph_rocs = []
ensemble_prcs = []
ensemble_rocs = []
for i in range(20):
    smiles, labels = smile_label_reader(seq_dir + 'results_bce_' + str(i) + '/valid.csv')
    seq_preds = np.load(seq_dir + 'results_bce_' + str(i) + '/valid_preds.npy')
    graph_preds = graph_preds_reader(graph_dir + 'ce_' + str(i) + '/val_preds.csv')
    
    seq_prc = prc_auc(labels, seq_preds)
    seq_roc = roc_auc_score(labels, seq_preds)
    print('seq_prc:', seq_prc, 'seq_roc:', seq_roc)
    graph_prc = prc_auc(labels, graph_preds)
    graph_roc = roc_auc_score(labels, graph_preds)
    print('graph_prc:', graph_prc, 'graph_roc:', graph_roc)
    

    ensemble_preds = (seq_preds[:,0] + graph_preds)/ 2
    ensemble_prc = prc_auc(labels, ensemble_preds)
    ensemble_roc = roc_auc_score(labels, ensemble_preds)
    print('ensemble_prc:', ensemble_prc, 'ensemble_roc:', ensemble_roc)
    print('==========================================')
    
    seq_prcs.append(seq_prc)
    seq_rocs.append(seq_roc)
    graph_prcs.append(graph_prc)
    graph_rocs.append(graph_roc)
    ensemble_prcs.append(ensemble_prc)
    ensemble_rocs.append(ensemble_roc)

seq_prc_mean = np.mean(seq_prcs)
seq_prc_std = np.std(seq_prcs)
seq_roc_mean = np.mean(seq_rocs)
seq_roc_std = np.std(seq_rocs)
print('Validation sequence PRC is: {:4f}+/-{:4f}'.format(seq_prc_mean, seq_prc_std))
print('Validation sequence ROC is: {:4f}+/-{:4f}'.format(seq_roc_mean, seq_roc_std))
print(' ')

graph_prc_mean = np.mean(graph_prcs)
graph_prc_std = np.std(graph_prcs)
graph_roc_mean = np.mean(graph_rocs)
graph_roc_std = np.std(ensemble_rocs)
print('Validation graph PRC is: {:4f}+/-{:4f}'.format(graph_prc_mean, graph_prc_std))
print('Validation graph ROC is: {:4f}+/-{:4f}'.format(graph_roc_mean, graph_roc_std))
print(' ')

ensemble_prc_mean = np.mean(ensemble_prcs)
ensemble_prc_std = np.std(ensemble_prcs)
ensemble_roc_mean = np.mean(ensemble_rocs)
ensemble_roc_std = np.std(ensemble_rocs)
print('Validation ensemble PRC is: {:4f}+/-{:4f}'.format(ensemble_prc_mean, ensemble_prc_std))
print('Validation ensemble ROC is: {:4f}+/-{:4f}'.format(ensemble_roc_mean, ensemble_roc_std))
print(' ')
    

