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





dataset_name = 'test'
sys.path.append("..")
confs = __import__('config.config_default', fromlist=['conf'])
conf = confs.conf


for i in range(20):
    
    out_path = '../results/atcc15579_oversample_10/ce_{}/'.format(i)
    print(out_path)

    ### Get and split dataset
    dataset, num_node_features, num_edge_features, num_graph_features = get_dataset('../../datasets/drug_pro_auprc/', dataset_name, conf['graph_level_feature'])


    ## Build model
    model = MLNet2(num_node_features, num_edge_features, num_graph_features, conf['hidden'], conf['dropout'], conf['num_tasks'], conf['depth'], conf['graph_level_feature'])
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(torch.load(out_path + 'params.ckpt'))
    
    
    ## Predict test dataset

    test_loader = DataLoader(dataset, conf['vt_batch_size'], shuffle=False) 
    save_preds(model, test_loader, device, out_path, 'test_preds')
    

