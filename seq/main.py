import os

from data import *
from train import Trainer
from config.train_config import conf_trainer, conf_tester
from auprc import *


os.environ['CUDA_VISIBLE_DEVICES'] = '4, 3'
conf_trainer['use_gpu'] = True
conf_tester['use_gpu'] = True

train_file = '../datasets/drug/atcc15579.csv'

for i in range(20): 
    train_smile, train_label, valid_smile, valid_label, _, _, train_smile_no_oversample, train_label_no_oversample = read_split_data(train_file, split_mode='pnrandom', split_ratios=[0.8, 0.2], seed=i, oversample=10)
    
    root_path1 = 'results/atcc15579_oversample_10/results_{}_{}/'.format(conf_trainer['loss'], i)
    if not os.path.isdir(root_path1):
        os.mkdir(root_path1)
        
    data_csv(train_smile_no_oversample, train_label_no_oversample, valid_smile, valid_label, root_path1)
    
    trainer = Trainer(conf_trainer, conf_tester, root_path1, train_smile, train_label, valid_smile, valid_label, None, None, train_smile_no_oversample, train_label_no_oversample)
    trainer.train()
    
    # save val_preds
    metrics = trainer.valider.multi_task_test(model_file=root_path1 + 'model.pth', npy_file=root_path1 + 'valid_preds.npy')
    print(metrics)
    



