import csv
from datasets import *

def data_reader(file_name):
    inputs = []
    labels = []
    with codecs.open(file_name, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = row['smiles']
            inputs.append(smiles)
            labels.append(
                [float(row[y]) if row[y] != '' else np.nan for y in row.keys() if y != 'smiles' and y != 'mol_id'])
        return inputs, np.array(labels)
    
smiles, labels = data_reader('../../datasets/drug/ecoli.csv')
dataset = [[smiles[i], int(labels[i][0])] for i in range(len(labels))]
dataset = shuffle(dataset, random_state=124)

split_size=[0.8, 0.2, 0.0]
train_size = int(split_size[0] * len(dataset))
train_val_size = int((split_size[0] + split_size[1]) * len(dataset))
train_dataset = dataset[:train_size]
val_dataset = dataset[train_size:train_val_size]
test_dataset = dataset[train_val_size:]


with open('train_s124.csv','w') as result_file:
    wr = csv.writer(result_file, dialect='excel')
    wr.writerow(['smiles', 'labels'])
    for data in train_dataset:
    	wr.writerow(data)

