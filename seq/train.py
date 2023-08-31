import numpy as np
import torch
from torch.utils.data import DataLoader
from metric import *
from models import *
import csv
import random
from data import *
from evaluate import Tester
import argparse
import os
import sys
from auprc import *



class Trainer():
    def __init__(self, conf_trainer, conf_tester, out_path, train_smile, train_label, valid_smile, valid_label, test_smile, test_label, train_smile_no_oversample, train_label_no_oversample):
        self.txtfile = os.path.join(out_path, 'record.txt')
        self.out_path = out_path
        self.config = conf_trainer

        batch_size, seq_max_len, use_aug, use_cls_token = self.config['batch_size'], self.config['seq_max_len'], self.config['use_aug'], self.config['use_cls_token']
        self.train_label = train_label
        self.trainset = TargetSet(train_smile, train_label, use_aug, use_cls_token, seq_max_len)
        self.trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.valider = Tester(valid_smile, valid_label, conf_tester)
        # self.tester = Tester(test_smile, test_label, conf_tester)
        # self.tester = Tester(test_smile, None, conf_tester)
        # self.test_label = test_label
        self.trainervalue = Tester(train_smile_no_oversample, train_label_no_oversample, conf_tester) #train_smile, train_label

        # seq_lens1 = np.max([len(x) for x in valid_smile])+80
        # seq_len = max(self.trainset.seq_len, seq_lens1)
        seq_lens1 = np.max([len(x) for x in valid_smile])+80
        # seq_lens2 = np.max([len(x) for x in test_smile])+80
        seq_lens2 = np.max([len(x) for x in valid_smile])+80
        seq_len = max(max(self.trainset.seq_len, seq_lens1), seq_lens2)
        self.net = self._get_net(self.trainset.vocab_size, seq_len=seq_len)
        if self.config['use_gpu']:
            self.net = torch.nn.DataParallel(self.net)
            self.net = self.net.to('cuda')

        self.criterion = self._get_loss_fn()
        self.optim = self._get_optim()
        self.lr_scheduler = self._get_lr_scheduler()

        self.start_epoch = 1
        if self.config['ckpt_file'] is not None:
            self.load_ckpt(self.config['ckpt_file'])

    def _get_net(self, vocab_size, seq_len):
        type, param = self.config['net']['type'], self.config['net']['param']
        if type == 'gru_att':
            model = GRU_ATT(vocab_size=vocab_size, **param)
            return model
        elif type == 'bert_tar':
            model = BERTChem_TAR(vocab_size=vocab_size, seq_len=seq_len, **param)
            if self.config['pretrain_model'] is not None:
                model.load_feat_net(self.config['pretrain_model'])
            if self.config['loss'] == 'auprc2':
                model.pred[0].linear.reset_parameters()
            return model
        else:
            raise ValueError('not supported network model!')
    
    def _get_loss_fn(self):
        type = self.config['loss']
        if type == 'bce':
            return bce_loss(use_gpu=self.config['use_gpu'])
        elif type == 'prc':
            return AUCPRHingeLoss()
        elif type == 'rank':
            return rank_loss()
        elif type == 'bce_rank':
            return bce_loss(use_gpu=self.config['use_gpu']), rank_loss()
        elif type == 'hinge':
            return hinge_loss()
        elif type == 'wb_bce':
            ratio = self.trainset.get_imblance_ratio()
            return bce_loss(weights=[1.0, ratio], use_gpu=self.config['use_gpu'])
        elif type == 'mask_bce':
            return masked_bce_loss()
        elif type == 'mse':
            return torch.nn.MSELoss(reduction='sum')
        elif type == 'auprc2':
            return None     
        else:
            raise ValueError('not supported loss function!')

    def _get_optim(self):
        type, param = self.config['optim']['type'], self.config['optim']['param']
        model_params = self.net.parameters()
        if type == 'adam':
            return torch.optim.Adam(model_params, **param)
        elif type == 'rms':
            return torch.optim.RMSprop(model_params, **param)
        elif type == 'sgd':
            return torch.optim.SGD(model_params, **param)
        else:
            raise ValueError('not supported optimizer!')

    def _get_lr_scheduler(self):
        type, param = self.config['lr_scheduler']['type'], self.config['lr_scheduler']['param']
        if type == 'linear':
            return LinearSche(self.config['epoches'], warm_up_end_lr=self.config['optim']['param']['lr'], **param)
        elif type == 'square':
            return SquareSche(self.config['epoches'], warm_up_end_lr=self.config['optim']['param']['lr'], **param)
        elif type == 'cos':
            return CosSche(self.config['epoches'], warm_up_end_lr=self.config['optim']['param']['lr'], **param)
        elif type is None:
            return None
        else:
            raise ValueError('not supported learning rate scheduler!')

    def _train_loss(self, dataloader):
        self.net.train()
        total_loss = 0
        for batch, data_batch in enumerate(dataloader):
            seq_inputs, lengths, labels = data_batch['seq_input'], data_batch['length'], data_batch['label']
            if self.config['use_gpu']:
                seq_inputs = seq_inputs.to('cuda')
                lengths = lengths.to('cuda')
                labels = labels.to('cuda')

            if self.config['net']['type'] in ['gru_att']:
                outputs = self.net(seq_inputs, lengths)
            elif self.config['net']['type'] in ['bert_tar']:
                outputs = self.net(seq_inputs)

            if self.config['loss'] in ['bce', 'wb_bce', 'rank', 'hinge']:
                loss = self.criterion(outputs, labels)
            elif self.config['loss'] in ['bce_rank']:
                loss1, loss2 = self.criterion[0](outputs, labels), self.criterion[1](outputs, labels)
                # print('\t loss1 {} loss2 {}'.format(loss1.to('cpu').item(), (0.1 * loss2).to('cpu').item()))
                loss = loss1 + 0.1 * loss2
            elif self.config['loss'] in ['prc']:
                logits = torch.log(outputs/(1-outputs))
                loss = self.criterion(logits, labels)
            elif self.config['loss'] in ['mse']:
                loss = self.criterion(outputs, labels) / outputs.shape[0]
            elif self.config['loss'] in ['mask_bce']:
                mask = data_batch['mask']
                if self.config['use_gpu']:
                    mask = mask.to('cuda')
                loss = self.criterion(outputs, labels, mask)
            elif self.config['loss'] in ['auprc2']:
                predScore = outputs
                g = pairLossAlg2(5, predScore[0], predScore[1:])
                p = calculateP(g, u, data_batch['idx'][0], 1)
                loss = surrLoss(g, p)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            total_loss += loss.to('cpu').item()
#             if batch % self.config['verbose'] == 0:
#                print('\t Training iteration {} | loss {}'.format(batch, loss.to('cpu').item()))

        print("\t Training loss {:.6f}".format(total_loss/(batch+1)), end = '')#,#average loss over batch, batch loss over samples

    def _valid(self, epoch, metric_name1=None, metric_name2=None):
        self.net.eval()
        metrics = self.valider.multi_task_test(model=self.net)
        # test_metrics = self.tester.multi_task_test(model=self.net, npy_file = os.path.join(self.out_path, 'pred_{}.npy'.format(epoch)))
        # pred = self.tester.multi_task_test(model=self.net, npy_file = os.path.join(self.out_path, 'pred_{}.npy'.format(epoch)))
        train_metrics = self.trainervalue.multi_task_test(model=self.net)
        # test_prcs = []
        # for label in self.test_label:
        #     prc, _ = compute_cls_metric(label[:,0], pred[:,0])
        #     test_prcs.append(prc)
        # if self.config['save_valid_records']:
        #     file_obj = open(self.txtfile, 'a')
        #     file_obj.write('Train | {:.4f} Validation | {:.4f} Test | {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(train_metrics[0], metrics[0], test_prcs[0], test_prcs[1], test_prcs[2], test_prcs[3]))
        #     file_obj.close()

        print(' epoch {:03d}: Train | prc {:.4f}, roc {:.4f}, Validation | prc {:.4f}, roc {:.4f}'.format(epoch, train_metrics[0], train_metrics[1], metrics[0], metrics[1]))
        fp = open(self.txtfile, 'a')
        fp.write(' epoch {:03d}: Train | prc {:.4f}, roc {:.4f}, Validation | prc {:.4f}, roc {:.4f}\n'.format(epoch, train_metrics[0], train_metrics[1], metrics[0], metrics[1]))
        fp.close()
        return metrics[0], metrics[1], train_metrics[0], train_metrics[1]
    
    def save_ckpt(self, epoch):
        net_dict = self.net.state_dict()
        checkpoint = {
            "net": net_dict,
            'optimizer': self.optim.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, os.path.join(self.out_path, 'ckpt_{}.pth'.format(epoch)))

    def load_ckpt(self, load_pth):
        checkpoint = torch.load(load_pth)
        self.net.load_state_dict(checkpoint['net'])
        self.optim.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch'] + 1

    def train(self):
        epoches, save_model = self.config['epoches'], self.config['save_model']
        #print("Initializing Training...")
        self.optim.zero_grad()
        
        if self.config['net']['param']['task'] == 'reg':
            best_metric1, best_metric2 = 1000, 1000
            metric_name1, metric_name2 = 'mae', 'rmse'
        else:
            best_metric1, best_metric2, best_ep_train_prc = 0, 0, 0
            metric_name1, metric_name2 = 'prc_auc', 'roc_auc'

        best_epoch = 0
        if self.config['loss'] == 'auprc2':
            global u
            u = torch.zeros([len(self.trainset)])    
        for i in range(self.start_epoch, epoches+1):
            #print("Epoch {} ...".format(i))
            if self.config['loss'] == 'auprc2':
                self.trainloader = DataLoader(self.trainset, batch_size=self.config['batch_size'], sampler=AUCPRSampler(self.train_label, self.config['batch_size']))
            self._train_loss(self.trainloader)    
            metric1, metric2, train_metric1, train_metric2 = self._valid(i, metric_name1, metric_name2)
            if save_model == 'best_valid':
                if (self.config['net']['param']['task'] == 'reg' and (best_metric2 > metric2)) or (self.config['net']['param']['task'] == 'cls' and (best_metric1 < metric1)):
                    #print('saving model...')
                    best_metric1, best_metric2, best_ep_train_prc, best_ep_train_roc = metric1, metric2, train_metric1, train_metric2
                    best_epoch = i
                    if self.config['use_gpu']:
                        self.net.module.save_model(os.path.join(self.out_path, 'model.pth'))
                    else:
                        self.net.save_model(os.path.join(self.out_path, 'model.pth'))
            # elif save_model == 'each':
            #     #print('saving model...')
            #     if self.config['use_gpu']:
            #         self.net.module.save_model(os.path.join(self.out_path, 'model_{}.pth'.format(i)))
            #     else:
            #         self.net.save_model(os.path.join(self.out_path, 'model_{}.pth'.format(i)))
            
            #if i % self.config['save_ckpt'] == 0:
            #    self.save_ckpt(i)
        print('best_epoch:',best_epoch, 'best_prc:', best_metric1)
        fp = open(self.txtfile, 'a')
        fp.write('best_epoch: {:03d}; Train prc: {:.4f}; Train roc: {:.4f}; Best val prc: {:.4f}; Best val roc: {:.4f}\n'.format(best_epoch, best_ep_train_prc, best_ep_train_roc, best_metric1, best_metric2))
        fp.close()


