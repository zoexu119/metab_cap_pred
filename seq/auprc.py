import torch
import numpy as np
from torch.utils.data.sampler import Sampler
import random



class AUCPRSampler(Sampler):

    def __init__(self, labels, batchSize, posNum=1):
        # positive class: minority class
        # negative class: majority class

        self.labels = labels
        self.posNum = posNum
        self.batchSize = batchSize

        self.clsLabelList = np.unique(labels)
        self.dataDict = {}

        for label in self.clsLabelList:
            self.dataDict[str(label)] = []

        for i in range(len(self.labels)):
            self.dataDict[str(self.labels[i][0])].append(i)

        self.ret = []


    def __iter__(self):
        minority_data_list = self.dataDict[str(1.0)]
        majority_data_list = self.dataDict[str(0.0)]

        # print(len(minority_data_list), len(majority_data_list))
        random.shuffle(minority_data_list)
        random.shuffle(majority_data_list)

        # In every iteration : sample 1(posNum) positive sample(s), and sample batchSize - 1(posNum) negative samples
        if len(minority_data_list) // self.posNum * (self.batchSize - self.posNum) >= len(majority_data_list): # At this case, we go over the all positive samples in every epoch.
            # extend the length of majority_data_list from  len(majority_data_list) to len(minority_data_list)* (batchSize-posNum)
            majority_data_list.extend(np.random.choice(majority_data_list, len(minority_data_list) // self.posNum * (self.batchSize - self.posNum) - len(majority_data_list), replace=True))

            for i in range(len(minority_data_list) // self.posNum):
                if self.posNum == 1:
                    self.ret.append(minority_data_list[i])
                else:
                    self.ret.extend(minority_data_list[i*self.posNum:(i+1)*self.posNum])

                startIndex = i*(self.batchSize - self.posNum)
                endIndex = (i+1)*(self.batchSize - self.posNum)
                self.ret.extend(majority_data_list[startIndex:endIndex])

        else: # At this case, we go over the all negative samples in every epoch.
            # extend the length of minority_data_list from len(minority_data_list) to len(majority_data_list)//(batchSize-posNum) + 1
            minority_data_list.extend(np.random.choice(minority_data_list, len(majority_data_list) // (self.batchSize - self.posNum) + 1 - len(minority_data_list), replace=True))

            for i in range(0, len(majority_data_list), self.batchSize - self.posNum):

                if self.posNum == 1:
                    self.ret.append(minority_data_list[i//(self.batchSize - self.posNum)])
                else:
                    self.ret.extend(minority_data_list[i//(self.batchSize- self.posNum)* self.posNum: (i//(self.batchSize-self.posNum) + 1)*self.posNum])

                self.ret.extend(majority_data_list[i:i + self.batchSize - self.posNum])
        return iter(self.ret)


    def __len__ (self):
        return len(self.ret)


def pairLossAlg1(temperature, f_p, f_n):

    f_p_vec = torch.ones_like(f_n) * f_p

    return torch.exp( (f_p_vec - f_n)/temperature)


def pairLossAlg2(threshold, f_p, f_n):

    # :param threshold: margin
    # :param f_p: prediction score of positive sample
    # :param f_n: prediction score of positive sample
    # :return: margin loss  g(w;\x_j,\x_s)

    f_p_vec = torch.ones_like(f_n) * f_p
    # print(f_p_vec.size(), f_n.size())
    return  torch.max(threshold - (f_p_vec -f_n), torch.zeros_like(f_p_vec))**2


def calculateP(loss, u, index_s, gamma):
    # return: \tilde{p_j} for eah j in the batch
    u[index_s] = (1-gamma) * u[index_s] + gamma * torch.mean(loss)
    p = loss/u[index_s]
    p.detach_()
    return p


def surrLoss(loss, p):

    # Return the surrogate loss, whose gradient is \sum\limits_j \tilde{p_j}\nabla g(w;\x_j,\x_s)

    return torch.sum(p*loss)