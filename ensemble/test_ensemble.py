import numpy as np
import csv
import codecs
import re
import os



def smile_reader(file_name):
    inputs = [] 
    with codecs.open(file_name, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = row['smiles']
            inputs.append(smiles)
        return inputs

def graph_preds_reader(file_name):
    preds = []
    with codecs.open(file_name, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred = float(row['preds'])
            preds.append(pred)
        return preds

def write_predictions(file_name, all_preds):
    transposed_preds = np.array(all_preds).T.tolist()
    fields = ['smiles']
    for i in range(20):
        name = 'seed_{}'.format(i)
        fields.append(name)
    fields.append('average')

    with open(file_name, 'w') as f:
        writer = csv.writer(f)     
        writer.writerow(fields)
        writer.writerows(transposed_preds)


name = 'atcc15579_oversample_10'
method = 'graph'
graph_dir = '../graph/results/{}/'.format(name)
## Use next two lines for seq model
# method = 'seq'
# seq_dir = '../seq/results/{}/'.format(name)
save_path = 'results_preds/{}/{}.csv'.format(name, method)
print(save_path)


save_list = []
smiles = smile_reader('../datasets/drug/test.csv')
save_list.append(smiles)

for i in range(20):
    if method == 'graph':
        preds = graph_preds_reader(graph_dir + 'ce_' + str(i) + '/test_preds.csv')
    else:
        preds = np.load(seq_dir + 'results_bce_' + str(i) + '/test_preds.npy')
        preds = np.squeeze(preds).tolist()
    save_list.append(preds)
    print('{} done'.format(i))

preds_list = np.array(save_list[1:])
avg_list = np.mean(preds_list, axis=0).tolist()
save_list.append(avg_list)
print('saving csv..')
write_predictions(save_path, save_list)
