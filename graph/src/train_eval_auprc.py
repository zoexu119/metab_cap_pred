import os
from numpy.lib import ufunclike
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader
from metric import compute_cla_metric
import numpy as np
from auprc import *
from loss import *
from auprc_hinge import *
import time
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### This is run function for classification tasks
def run_classification(train_dataset, val_dataset, test_dataset, model, num_tasks, epochs, batch_size, vt_batch_size, lr,
        lr_decay_factor, lr_decay_step_size, weight_decay, loss_type='ce', loss_param={}, ft_mode='fc_random', pre_train=None, save_dir=None):

    model = model.to(device)
    if pre_train is not None:
        model.load_state_dict(torch.load(pre_train))
    if ft_mode == 'fc_random':
        model.mlp1.reset_parameters()

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    global u, a, b, m, alpha, lamda
    if loss_type == 'ce':
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    elif loss_type in ['auprc1', 'auprc2']:
        labels = [int(data.y.item()) for data in train_dataset]
        criterion = None       
        u = torch.zeros([len(train_dataset)+len(val_dataset)+len(test_dataset)])
    elif loss_type in ['sum']:
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        labels = [int(data.y.item()) for data in train_dataset]
        u = torch.zeros([len(train_dataset)+len(val_dataset)+len(test_dataset)])
    elif loss_type in ['wce','focal','ldam']:
        labels = [int(data.y.item()) for data in train_dataset]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        cls_num_list = [n_neg, n_pos]
        if loss_type == 'wce':
            criterion = WeightedBCEWithLogitsLoss(cls_num_list=cls_num_list)
        elif loss_type == 'focal':
            criterion = FocalLoss(cls_num_list=cls_num_list)
        elif loss_type == 'ldam':
            criterion = BINARY_LDAMLoss(cls_num_list=cls_num_list)
    elif loss_type in ['auroc', 'auroc2']:
        criterion = None
        a, b, alpha, m = float(1), float(0), float(1), loss_param['m']
        labels = [int(data.y.item()) for data in train_dataset]
        loss_param['pos_ratio'] = sum(labels) / len(labels)
    elif loss_type in ['auprc_lang']:
        # criterion = None
        # lamda = torch.zeros(loss_param['K'], dtype=torch.float32, device='cuda', requires_grad=True).cuda()
        # # lamda.data += torch.tensor(list(range(1, 101, 11)), device='cuda') * 1.0 / 100
        # lamda.data += 0.5
        # b = torch.zeros(loss_param['K'], dtype=torch.float32, device='cuda', requires_grad=True).cuda()
        # # b.data += torch.tensor(list(range(1, 101, 11)), device='cuda') * 1.0 / 100
        # b.data += 1.0

        # labels = [int(data.y.item()) for data in train_dataset]
        # loss_param['pos_ratio'] = sum(labels) / len(labels)
        criterion = AUCPRHingeLoss()
    
    train_loader_for_prc = DataLoader(train_dataset, vt_batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, vt_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False)
    
    best_val_metric = 0
    best_val_roc = 0
    best_test_metric = 0
    best_ep_train_prc = 0
    best_ep_train_roc = 0
    epoch_bvl = 0
    epoch_test = 0

    start = time.time()
    for epoch in range(1, epochs + 1):
        if loss_type in ['ce', 'wce', 'focal', 'ldam', 'auroc', 'auroc2', 'auprc_lang']:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
        elif loss_type in ['auprc1', 'auprc2', 'sum']:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=AUCPRSampler(labels, batch_size))
        
        avg_train_loss = train_classification(model, optimizer, train_loader, num_tasks, device, epoch, lr, criterion, loss_type, loss_param)
        train_prc_results, train_roc_results = test_classification(model, train_loader_for_prc, num_tasks, device)
        val_prc_results, val_roc_results = test_classification(model, val_loader, num_tasks, device)
        if len(test_dataset) > 0:
            test_prc_results, test_roc_results = test_classification(model, test_loader, num_tasks, device)
        else:
            test_prc_results, test_roc_results = -1, -1

            # print('Epoch: {:03d}, Training Loss: {:.6f}, Val PRC (avg over multitasks): {:.4f}'.format(epoch, avg_train_loss, np.mean(val_prc_results)))

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

        if np.mean(val_prc_results) > best_val_metric:
            epoch_bvl = epoch
            best_val_metric = np.mean(val_prc_results)
            best_val_roc = np.mean(val_roc_results)
            best_ep_train_prc = np.mean(train_prc_results)
            best_ep_train_roc = np.mean(train_roc_results)
            torch.save(model.state_dict(), os.path.join(save_dir, 'params.ckpt'.format(epoch)))
        if np.mean(test_prc_results) > best_test_metric:
            epoch_test = epoch
            best_test_metric = np.mean(test_prc_results)
        
        if save_dir is not None:
#             torch.save(model.state_dict(), os.path.join(save_dir, 'params{}.ckpt'.format(epoch)))
            fp = open(os.path.join(save_dir, 'record.txt'), 'a')
            fp.write('Epoch: {:03d}, Train avg loss: {:.4f}, Train PRC: {:.4f}, Train ROC: {:.4f}, Val PRC: {:.4f}, Val ROC: {:.4f}, Test PRC: {:.4f}\n'.format(epoch, avg_train_loss, np.mean(train_prc_results), np.mean(train_roc_results), np.mean(val_prc_results), np.mean(val_roc_results), np.mean(test_prc_results)))
            fp.close()
#             print('Epoch: {:03d}, Train avg loss: {:.4f}, Train PRC: {:.4f}, Val PRC: {:.4f}, Test PRC: {:.4f}\n'.format(epoch, avg_train_loss, np.mean(train_prc_results), np.mean(val_prc_results), np.mean(test_prc_results)))

    end = time.time()
    runtime = end - start
    fp = open(os.path.join(save_dir, 'record.txt'), 'a')
    fp.write('Train prc: {:.4f}, Train roc: {:.4f}, Best val prc: {:.4f}, Best val roc: {:.4f}, Best val metric achieves at epoch: {:03d}\n'.format(best_ep_train_prc, best_ep_train_roc, best_val_metric, best_val_roc, epoch_bvl))
    fp.close()
#     print('Best val metric is: {:.4f}, Best val metric achieves at epoch: {:03d}\n'.format(best_val_metric, epoch_bvl))
#     print('Best test metric is: {:.4f}, Best test metric achieves at epoch: {:03d}\n'.format(best_test_metric, epoch_test))
#     print('Running time is {:.4f}\n'.format(runtime))

    
def train_classification(model, optimizer, train_loader, num_tasks, device, epoch, lr, criterion=None, loss_type=None, loss_param={}):
    model.train()
    
    global a, b, m, alpha
    if loss_type == 'auroc2' and epoch % 10 == 1:
        # Periordically update w_{ref}, a_{ref}, b_{ref}
        global state, a_0, b_0
        a_0, b_0 = a, b
        state = []
        for name, param in model.named_parameters():
            state.append(param.data)

    losses = []
    for i,batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        batch_data = batch_data.to(device)
        out = model(batch_data)

        if loss_type == 'ce':
            if len(batch_data.y.shape) != 2:
                batch_data.y = torch.reshape(batch_data.y, (-1, num_tasks))
            mask = torch.Tensor([[not np.isnan(x) for x in tb] for tb in batch_data.y.cpu()]) # Skip those without targets (in PCBA, MUV, Tox21, ToxCast)
            mask = mask.to(device)
            target = torch.Tensor([[0 if np.isnan(x) else x for x in tb] for tb in batch_data.y.cpu()])
            target = target.to(device)
            loss = criterion(out, target) * mask
            loss = loss.sum()
            loss.backward()
            optimizer.step()
        elif loss_type == 'auprc1':
            target = batch_data.y
            predScore = torch.nn.Sigmoid()(out)
            g = pairLossAlg1(loss_param['temp'], predScore[0], predScore[1:])
            p = calculateP(g, u, batch_data.idx[0], 1)
            loss = surrLoss(g, p)
            loss.backward()
            optimizer.step()
        elif loss_type == 'auprc2':
            target = batch_data.y
            predScore = torch.nn.Sigmoid()(out)
            g = pairLossAlg2(loss_param['threshold'], predScore[0], predScore[1:])
            p = calculateP(g, u, batch_data.idx[0], 1)
            loss = surrLoss(g, p)
            loss.backward()
            optimizer.step()
        elif loss_type == 'sum':
            target = batch_data.y
            if len(target.shape) != 2:
                target = torch.reshape(target, (-1, num_tasks))
            loss1 = criterion(out, target)
            loss1 = loss1.sum()
            predScore = torch.nn.Sigmoid()(out)
            g = pairLossAlg2(10, predScore[0], predScore[1:])
            p = calculateP(g, u, batch_data.idx[0], 1)
            loss2 = surrLoss(g, p)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
        elif loss_type in ['wce','focal','ldam']:
            target = batch_data.y
            loss = criterion(out, target, epoch)
            loss.backward()
            optimizer.step()
        elif loss_type in ['auroc']:
            target = batch_data.y
            predScore = torch.nn.Sigmoid()(out)
            loss = AUROC_loss(predScore, target, a, b, m, alpha, loss_param['pos_ratio'])
            a, b, alpha = PESG_update_a_b_alpha(lr, a, b, alpha, m, predScore, target, loss_param['pos_ratio'])
            loss.backward()
            optimizer.step()
        elif loss_type in ['auroc2']:
            target = batch_data.y
            predScore = torch.nn.Sigmoid()(out)
            loss = AUROC_loss(predScore, target, a, b, m, alpha, loss_param['pos_ratio'])

            curRegularizer = calculateRegularizerWeights(lr, model, state, loss_param['gamma'])
            loss.backward()
            optimizer.step()
            regularizeUpdate(model, curRegularizer)
            a, b, alpha = PESG_update_a_b_alpha_2(lr, a, a_0, b, b_0, alpha, m, predScore, target, loss_param['pos_ratio'], loss_param['gamma'])
        elif loss_type in ['auprc_lang']:
            # target = batch_data.y
            # predScore = torch.nn.Sigmoid()(out[:,0])
            # loss = AUPRCWithLang(predScore, target, loss_param['K'], b, lamda, loss_param['pos_ratio'])

            # model.zero_grad()
            # b.grad = None
            # lamda.grad = None
            # loss.backward()

            # for _, param in model.named_parameters():
            #     if param.grad is not None:
            #         param.data = param.data - lr * param.grad
            # b.data = b.data - lr * b.grad
            # lamda.data = lamda.data + lr * lamda.grad
            # lamda.data[lamda.data <= 0] = 0

            target = batch_data.y
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            
        # print('Iter {} | Loss {}'.format(i, loss.cpu().item()))
        losses.append(loss)
    return sum(losses).item() / len(losses)



def test_classification(model, test_loader, num_tasks, device):
    model.eval()

    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)
    for batch_data in test_loader:
        batch_data = batch_data.to(device)
        with torch.no_grad():
            out = model(batch_data)
        if len(batch_data.y.shape) != 2:
            batch_data.y = torch.reshape(batch_data.y, (-1, num_tasks))
        pred = torch.sigmoid(out) ### prediction real number between (0,1)
        preds = torch.cat([preds,pred], dim=0)
        targets = torch.cat([targets, batch_data.y], dim=0)
    prc_results, roc_results = compute_cla_metric(targets.cpu().detach().numpy(), preds.cpu().detach().numpy(), num_tasks)
    
    return prc_results, roc_results

def save_preds(model, test_loader, device, save_dir, save_name='val_preds'):
    model.eval()

    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)
    for batch_data in test_loader:
        batch_data = batch_data.to(device)
        with torch.no_grad():
            out = model(batch_data)
        if len(batch_data.y.shape) != 2:
            batch_data.y = torch.reshape(batch_data.y, (-1, 1))
        pred = torch.sigmoid(out) ### prediction real number between (0,1)
        preds = torch.cat([preds,pred], dim=0)
        targets = torch.cat([targets, batch_data.y], dim=0)
    
    datas = [[preds[i].item(), int(targets[i])] for i in range(len(preds))]
        
    with open(save_dir + save_name+'.csv','w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(['preds', 'labels'])
        for data in datas:
            wr.writerow(data)

