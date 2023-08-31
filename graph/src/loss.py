import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sigmoidf = nn.Sigmoid()


def get_weight(epoch, beta=0.9999, cls_num_list=[]):
    '''

    :param args:
    :param epoch:
    :return: The weights for positive and negative weights
    '''
    per_cls_weights = None
    if epoch <= 10:
        per_cls_weights = np.ones([2])
    else:
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    # per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
    
    return per_cls_weights


def focal_loss(predScore, target, alpha, gamma, weight):
    """Computes the focal loss"""

    '''
    input_values = -\log(p_t)
    loss = - \alpha_t (1-\p_t)\log(p_t)
    '''
    loss = torch.zeros_like(predScore)
    if len(loss[target == 1])!=0:
        loss[target == 1] = -alpha * (1 - predScore[target == 1]) ** gamma *torch.log(predScore[target == 1]) * weight[target == 1]
    if len(loss[target == 0]) != 0:
        loss[target == 0] = -alpha * (predScore[target == 0]) ** gamma *torch.log(1- predScore[target == 0]) * weight[target == 0]

    return loss.mean()


class FocalLoss(nn.Module):
    '''
    Input of loss is the output of the model
    '''
    def __init__(self, cls_num_list=[], alpha = 1, gamma=0, reduction = 'mean'):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.cls_num_list = cls_num_list
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, input, target, epoch):
        weight = get_weight(epoch, cls_num_list=self.cls_num_list)
        indexTarget=target.detach().cpu().numpy()
        predScore = sigmoidf(input)

        return focal_loss(predScore, target, self.alpha, self.gamma, weight[indexTarget].view(-1,1))


class BINARY_LDAMLoss(nn.Module):
    def __init__(self, cls_num_list=[], max_m=0.5, s=30, reduction='mean'):
        super(BINARY_LDAMLoss, self).__init__()
        self.cls_num_list = cls_num_list
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.reduction = reduction

    def forward(self, outputs, target, epoch):
        indexWeight = target.detach().cpu().numpy()
        margin_outputs = outputs-self.m_list[indexWeight].view(-1,1) # margin of the output
        weight = get_weight(epoch, cls_num_list=self.cls_num_list)

        return F.binary_cross_entropy_with_logits(self.s * margin_outputs, 1.0*target.view(-1,1), weight=weight[indexWeight].view(-1,1), reduction=self.reduction)


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, cls_num_list=[], reduction = 'mean'):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.cls_num_list = cls_num_list
        self.reduction = reduction

    def forward(self, input, target, epoch):
        indexTarget=target.detach().cpu().numpy()
        weight = get_weight(epoch, cls_num_list=self.cls_num_list)

        return F.binary_cross_entropy_with_logits(input, 1.0*target.view(-1,1), weight=weight[indexTarget].view(-1,1))


def AUROC_loss(predSocre, targets, a, b, m, alpha, p_hat=0.035):
    '''
    input is prediction score
    m is tunable (1), a, b, alpha are trainable variable that initialized by 1, 0, 1 respectively
    alpha >= 0
    reference: https://arxiv.org/abs/2012.03173
    '''

    comp1 = (1 - p_hat) * torch.mean((predSocre.view(-1) - a) ** 2 * (1 == targets).float())
    comp2 = p_hat * torch.mean((predSocre.view(-1)  - b) ** 2 * (0 == targets).float())
    comp3 = 2 * alpha * (p_hat * (1 - p_hat) * m + torch.mean( p_hat * predSocre.view(-1)  * (0 == targets).float() - (1 - p_hat) * predSocre.view(-1) * (1 == targets).float()))
    comp4 = p_hat * (1 - p_hat) * (alpha ** 2)
    loss = comp1 + comp2 + comp3 - comp4

    return loss


def PESG_update_a_b_alpha(lr, a, b, alpha, m, predScore, targets, pos_ratio=0.035):
    grad_a, grad_b = 0, 0
    posNum = torch.sum((1==targets).float())
    negNum = torch.sum((0==targets).float())

    grad_a = -2*(1-pos_ratio)*torch.sum((predScore.view(-1) - a)*(1==targets).float())/(posNum+negNum)
    grad_b = -2*pos_ratio*torch.mean((predScore.view(-1)-b)*(0==targets).float())/(posNum+negNum)
    aver_neg = torch.sum(pos_ratio * predScore.view(-1) * (0 == targets).float())/(posNum+negNum)
    aver_pos = torch.sum((1 - pos_ratio) * predScore.view(-1) * (1 == targets).float()) / (posNum + negNum)

    grad_alpha = -2*pos_ratio*(1-pos_ratio)*alpha+2*(pos_ratio*(1-pos_ratio)*m + aver_neg - aver_pos)

    a = a -lr * grad_a
    b = b -lr * grad_b
    alpha = alpha + lr * grad_alpha
    if alpha <= 0:
        alpha = torch.zeros(1)

    return a.item(),b.item(), alpha.item()


def PESG_update_a_b_alpha_2(lr, a, a_0, b, b_0, alpha, m, predScore, targets, pos_ratio=0.035, gamma=1000):
    grad_a, grad_b = 0, 0
    posNum = torch.sum((1==targets).float())
    negNum = torch.sum((0==targets).float())

    grad_a = -2*(1-pos_ratio)*torch.sum((predScore.view(-1) - a)*(1==targets).float())/(posNum+negNum)
    grad_b = -2*pos_ratio*torch.mean((predScore.view(-1)-b)*(0==targets).float())/(posNum+negNum)
    aver_neg = torch.sum(pos_ratio * predScore.view(-1) * (0 == targets).float())/(posNum+negNum)
    aver_pos = torch.sum((1 - pos_ratio) * predScore.view(-1) * (1 == targets).float()) / (posNum + negNum)

    grad_alpha = -2*pos_ratio*(1-pos_ratio)*alpha+2*(pos_ratio*(1-pos_ratio)*m + aver_neg - aver_pos)

    a = a - lr * (grad_a + 1/gamma *(a-a_0))
    b = b - lr * (grad_b + 1/gamma *(b-b_0))
    alpha = alpha + lr * grad_alpha
    if alpha <= 0:
        alpha = torch.zeros(1)

    return a.item(),b.item(), alpha.item()


def calculateRegularizerWeights(lr, model, state, gamma=1000):

    res = []
    k = 0
    for name, param in model.named_parameters():

        weight = lr * (param.data - state[k])/gamma
        k += 1
        res.append(weight)
    return res


def regularizeUpdate(model, curRegularizer):
    k = 0
    for name, param in model.named_parameters():
        param.data = param.data - curRegularizer[k]
        k += 1