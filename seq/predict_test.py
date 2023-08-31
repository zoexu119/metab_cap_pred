import os

from data import *
from train import Trainer
from config.train_config import conf_trainer, conf_tester
from auprc import *


os.environ['CUDA_VISIBLE_DEVICES'] = '2, 7'
conf_trainer['use_gpu'] = True
conf_tester['use_gpu'] = True

test_file = '../datasets/drug/test.csv'

for i in range(20): 
    test_smile, pseudo_label = read_split_data(test_file, split_mode=None)

    root_path1 = 'results/atcc15579_oversample_10/results_{}_{}/'.format(conf_trainer['loss'], i)
    print(root_path1)

    trainer = Trainer(conf_trainer, conf_tester, root_path1, test_smile, pseudo_label, test_smile, None, None, None, None, None)

    # save val_preds
    preds = trainer.valider.multi_task_test(model_file=root_path1 + 'model.pth', npy_file=root_path1 + 'test_preds.npy')

    



