from sklearn.metrics import auc,  precision_recall_curve, roc_auc_score, mean_absolute_error, mean_squared_error
import numpy as np
import scipy.stats as ss



def prc_auc(targets, preds):
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)




def compute_cla_metric(targets, preds, num_tasks):
    
    prc_results = []
    roc_results = []
    for i in range(num_tasks):
        is_labeled = targets[:,i] == targets[:,i] ## filter some samples without groundtruth label
        target = targets[is_labeled,i]
        pred = preds[is_labeled,i]
        try:
            prc = prc_auc(target, pred)
        except ValueError:
            prc = np.nan
            print("In task #", i+1, " , there is only one class present in the set. PRC is not defined in this case.")
        try:
            roc = roc_auc_score(target, pred)
        except ValueError:
            roc = np.nan
            print("In task #", i+1, " , there is only one class present in the set. ROC is not defined in this case.")
        if not np.isnan(prc): 
            prc_results.append(prc)
        else:
            print("PRC results do not consider task #", i+1)
        if not np.isnan(roc): 
            roc_results.append(roc)
        else:
            print("ROC results do not consider task #", i+1)

        ## check rank of positive samples
        rank_pred = len(pred) - ss.rankdata(pred).astype(int)
        pos_rank = rank_pred[target == 1]

    return prc_results, roc_results
    # return prc_results, roc_results, pos_rank




def compute_reg_metric(targets, preds, num_tasks):
    mae_results = []
    rmse_results = []
    for i in range(num_tasks):
        target = targets[:,i]
        pred = preds[:,i]
        mae = mean_absolute_error(target, pred)
        rmse = np.sqrt(mean_squared_error(target, pred))
        mae_results.append(mae)
        rmse_results.append(rmse)
    return mae_results, rmse_results
    