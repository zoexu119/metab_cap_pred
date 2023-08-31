import argparse
import torch
import torch.nn.functional as F
from texttable import Texttable
import sys

from datasets import *
# from train_eval_orig import run_classification, run_regression
from train_eval_auprc import *
from torch_scatter import scatter_mean
from model import *
import os
from torch_geometric.data import DataLoader
from train_eval_auprc import test_classification





dataset_name = 'ntcc9343'
### Use corresponding configuration for different datasets
sys.path.append("..")
confs = __import__('config.config_default', fromlist=['conf'])
conf = confs.conf
# print(conf)


for i in range(20):
    
    out_path = '../results/atcc15579_oversample_10/ce_{}/'.format(i)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    txtfile = out_path+'record.txt'

    print(conf)
    file_obj = open(txtfile, 'w')
    file_obj.write('================\n')
    file_obj.write('conf: {}\n'.format(conf))
    file_obj.write('================\n')
    file_obj.close()

    ### Get and split dataset

    dataset, num_node_features, num_edge_features, num_graph_features = get_dataset('../../datasets/drug_pro_auprc/', dataset_name, conf['graph_level_feature'])

    train_dataset, val_dataset, test_dataset = split_data('../../datasets/drug/', dataset_name, dataset, 'pnrandom', seed=i, split_size=[0.8, 0.2, 0.0], oversample=10)

    train_pos = torch.sum(torch.tensor([data.y for data in train_dataset]))
    val_pos = torch.sum(torch.tensor([data.y for data in val_dataset]))
    test_pos = torch.sum(torch.tensor([data.y for data in test_dataset]))

    print("======================================")
    print("=====Total number of graphs in", dataset_name,":", len(train_dataset)+len(val_dataset)+len(test_dataset), "=====")
    print("=====Total number of training graphs in", dataset_name,":", len(train_dataset), "=====")
    print("=====Total number of validation graphs in", dataset_name,":", len(val_dataset), "=====")
    print("=====Total number of test graphs in", dataset_name,":", len(test_dataset), "=====")
    print('=====Number of positive samples in training set:', int(train_pos.item()))
    print('=====Number of positive samples in vailidation set:', int(val_pos.item()))
    print('=====Number of positive samples in test set:', int(test_pos.item()))
    print("======================================")


    ## Build model
    model = MLNet2(num_node_features, num_edge_features, num_graph_features, conf['hidden'], conf['dropout'], conf['num_tasks'], conf['depth'], conf['graph_level_feature'])

    ## Run
    run_classification(train_dataset, val_dataset, test_dataset, model, conf['num_tasks'], conf['epochs'], conf['batch_size'], conf['vt_batch_size'], conf['lr'], conf['lr_decay_factor'], conf['lr_decay_step_size'], conf['weight_decay'], conf['loss_type'], conf['loss_param'], conf['ft_mode'], conf['pre_train'], out_path)
    
    ## Predict val_dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(torch.load(out_path + 'params.ckpt'))
 
    train_loader_for_prc = DataLoader(train_dataset, conf['vt_batch_size'], shuffle=False)
    val_loader = DataLoader(val_dataset, conf['vt_batch_size'], shuffle=False)
    
    save_preds(model, val_loader, device, out_path)
        
    train_prc_results, train_roc_results = test_classification(model, train_loader_for_prc, 1, device)
    val_prc_results, val_roc_results = test_classification(model, val_loader, 1, device)
    
    print(train_prc_results, train_roc_results)
    print(val_prc_results, val_roc_results)
