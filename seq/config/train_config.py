"""
    This is the configuration file for fine-tuning.
    We implemented two pretrain tasks: mask prediction and mask contrastive learning.
    --- Mask prediction is just the same pretrain task used in original BERT paper, predicting the ids of masked tokens
    --- Mask contrastive learning is our propsed pretraining task.  
"""

######################################################################################################################
# Settings for BERT network
## task: 'reg' for regression and 'cls' for classification
## n_out: number of outputs, it is the same as the number of labels of each training sample, or number of tasks
## hidden: number of hidden units in each Transformer layer
## n_layers: number of Transformer layers
## attn_heads: number of attention heads in each Transformer layer
## dropout: dropout rate in each dropout layer
## activation: the non-linear activation function used in feed-forward network of each Transformer layer, use 'relu' or 'gelu'
######################################################################################################################
conf_net = {}
conf_net['type'] = 'bert_tar'
#conf_net['param'] = {'task':'reg', 'n_out':12, 'hidden':1024, 'n_layers':6, 'attn_heads':4, 'dropout':0.1, 'activation':'gelu'}
conf_net['param'] = {'task':'cls', 'n_out':1, 'hidden':1024, 'n_layers':6, 'attn_heads':4, 'dropout':0.1, 'activation':'gelu'}

######################################################################################################################
# Setting for optimizer
## Type can be 'adam' (torch.optim.Adam), 'rms' (torch.optim.RMSprop), or 'sgd' (torch.optim.SGD)
## Param is just the parameters of optimizer, the parameter names are the same as those in pytorch
######################################################################################################################
conf_optim = {}
conf_optim['type'] = 'adam'
conf_optim['param'] = {'betas':(0.9,0.999), 'weight_decay':0, 'lr': 1e-5} #2e-5 1e-2 1e-4 1e-6

######################################################################################################################
# Setting for loss function
## We implement four types of loss function:
## 'mse': minimum square error loss, used for regression task
## 'bce': binary cross entropy loss, used for classification task
## 'wb_bce': weight-balanced binary cross entropy loss, used for classification task (can only be used in datasets with one task), use weights
##                      that are inversely propotional to the number of positive (negative) samples in training set for the loss of positive (negative) samples
## 'mask_bce': specially designed for classification datasets with missing labels (e.g. pcba, muv, tox21 and toxcast in MoleculeNet), do not
##                           compute loss for missing labels
######################################################################################################################
conf_loss = 'bce'#'prc'

######################################################################################################################
# Setting for learning rate scheduler
##  We provide three types of learning scheduler:
## --- 'cos' Cosine Annealing
## --- 'linear' Learing rate is decayed linearly
## --- 'square' Learing rate is inversely proportional to the square root of current epoch num
## --- If you set 'type' as None, no learning rate scheduler will be used, i.e. learning rate is a constant
## Learning rate is linearly increased from 'init_lr' (conf_lr_scheduler['param']['init_lr'] to 'base_lr' (the 'lr' in conf_optim['param']), i.e
## warming up, in the first several epoches setted by 'warm_up_epoches' (set 'warm_up_epoches' as 1 to skip warming up), then it is decayed.
## The decaying type is different for different learning rate scheduler.
######################################################################################################################
conf_lr_scheduler = {}
conf_lr_scheduler['type'] = None
conf_lr_scheduler['param'] = None

conf_trainer = {}
conf_trainer['optim'] = conf_optim
conf_trainer['loss'] = conf_loss
conf_trainer['lr_scheduler'] = conf_lr_scheduler
conf_trainer['net'] = conf_net

######################################################################################################################
# Other settings for training
## epoches: number of epoches
## batch_size: batch size
## seq_max_len: if it is not None, the dataloader will automatically filter the sequences with a length longer than it. It can help save memory.
## verbose: after how many iterations the loss value will be displayed in terminal
## save_ckpt: after how many epoches the current checkpoint will be saved
## ckpt_file: checkpoint file path, if it is not None, the program will start training from this saved checkpoint
## use_aug: whether data augmentation will be applied in training
## use_cls_token: set it to True when using BERT model
## pretrain_model: pretrain model file path, if it is not None, the model is initialized with the parameters of this pretrain model
## 'save_mode': if it is 'each', then the model will be saved in each epoch; if it is 'best_valid', then only the model achieving the best validation
##                              performance will be saved
## 'save_valid_records': if it is True, then the validation result will be recorded in each epoch
######################################################################################################################
conf_trainer['epoches'] = 200
conf_trainer['batch_size'] = 64
conf_trainer['seq_max_len'] = None
conf_trainer['verbose'] = 1
conf_trainer['save_ckpt'] = 5
conf_trainer['ckpt_file'] = None
conf_trainer['use_aug'] = True 
conf_trainer['use_cls_token'] = True
conf_trainer['pretrain_model'] = '../pretrain_model/mask_con.pth' #None#
conf_trainer['save_model'] = 'best_valid'
conf_trainer['save_valid_records'] = True

######################################################################################################################
# Other settings for validation/test
## batch_size: batch size
## task: 'reg' for regression and 'cls' for classification
## use_aug: whether data augmentation will be applied in training
## use_cls_token: set it to True when using BERT model
######################################################################################################################
conf_tester = {}
conf_tester['loss'] = conf_loss
conf_tester['net'] = conf_net
conf_tester['batch_size'] =32
conf_tester['task'] = 'cls'
conf_tester['use_aug'] = False #True
conf_tester['use_cls_token'] = True

assert conf_optim['type'] in ['adam', 'rms', 'sgd']
assert conf_lr_scheduler['type'] in ['cos', 'linear', 'square', None]
assert conf_net['param']['task'] == conf_tester['task']