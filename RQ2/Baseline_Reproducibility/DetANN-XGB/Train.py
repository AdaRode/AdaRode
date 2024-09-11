import numpy as np
import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Define the padding function
def pad_sequences(sequences, maxlen, padding='post'):
    padded_sequences = np.zeros((len(sequences), maxlen))
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            if padding == 'post':
                padded_sequences[i] = seq[:maxlen]
            elif padding == 'pre':
                padded_sequences[i] = seq[-maxlen:]
        else:
            if padding == 'post':
                padded_sequences[i, :len(seq)] = seq
            elif padding == 'pre':
                padded_sequences[i, -len(seq):] = seq
    return padded_sequences

class DataProcessor:
    def __init__(self, file_path, verbose=1, seed=1):
        self.file_path = file_path
        self.verbose = verbose
        self.seed = seed

    def read_data(self):
        if self.verbose > 0:
            print("Reading data from file:", self.file_path)

    def get_data(self):
        if self.verbose > 0:
            print("Getting data from file:", self.file_path)

class Action_json_data(DataProcessor):

    def __init__(self, file_path, verbose=1, seed=1):
        super().__init__(file_path=file_path, verbose=verbose, seed=seed)

    def _read_data(self):
        if self.verbose > 0:
            print("Reading data from file:", self.file_path)
        data = pd.read_csv(self.file_path, encoding="utf-8")
        return data

    def get_data(self) -> dict:
        seed = self.seed
        json_data = self._read_data()
        texts = json_data['Text']
        labels = json_data['Label']
        type_injections = json_data['type']

        res = {
            "texts": texts,
            "labels": labels,
            "types": type_injections
        }
        return res

def extract_features(payload, keywords, keyword_weights):
    lp = len(payload)
    nk = sum(payload.count(k) for k in keywords)
    kws = sum(keyword_weights.get(k, 0) * payload.count(k) for k in keywords)
    nspa = payload.count(' ') + payload.count('%20')
    rspa = nspa / lp if lp > 0 else 0
    nspe = sum(1 for c in payload if c in "!@#$%^&*()-_=+[]{}|;:'\",<.>/?")
    rspe = nspe / lp if lp > 0 else 0
    psna = nspa / lp if lp > 0 else 0
    psne = nspe / lp if lp > 0 else 0
    roc = 1 - psna - psne
    
    return [lp, nk, kws, nspa, rspa, nspe, rspe, roc]

keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'UNION', 'OR', 'AND', 'FROM', 'WHERE']
keyword_weights = {k: 1 for k in keywords}

def partition_data(data_list_pad, data_list_id, data_list_label, split_seed, testsize=0.2, validationsize=0.2, usetestflag=False):
    if not usetestflag:
        train_vali_set_x, test_set_x, train_vali_set_y, test_set_y, train_vali_set_id, test_set_id = train_test_split(
            data_list_pad, data_list_label, data_list_id, test_size=testsize, random_state=split_seed)
        train_set_x, validation_set_x, train_set_y, validation_set_y, train_set_id, validation_set_id = train_test_split(
            train_vali_set_x, train_vali_set_y, train_vali_set_id, test_size=validationsize, random_state=split_seed)
        return train_set_x, train_set_y, train_set_id, validation_set_x, validation_set_y, validation_set_id, test_set_x, test_set_y, test_set_id
    else:
        train_set_x, validation_set_x, train_set_y, validation_set_y, train_set_id, validation_set_id = train_test_split(
            data_list_pad, data_list_label, data_list_id, test_size=testsize, random_state=split_seed)
        return train_set_x, train_set_y, train_set_id, validation_set_x, validation_set_y, validation_set_id

config = yaml.safe_load(open("./TFModel/config/smallmodel.yaml", 'r', encoding="UTF-8"))

dataset = config.get('SmallModel').get('dataset')
embedding_method = config.get('SmallModel').get('embedding_method')
max_sequence_length = config.get('SmallModel').get('max_sequence_length')
split_seed = config.get('SmallModel').get('split_seed')
tokenizer_saved_path = config.get('SmallModel').get('tokenizer_saved_path')
embedding_saved_path = config.get('SmallModel').get('embedding_saved_path')
model_save_path = config.get('SmallModel').get('model_save_path')
epochs = config.get('SmallModel').get('epochs')
patience = config.get('SmallModel').get('patience')

datapath = os.getcwd() + os.sep + "Data" + os.sep + dataset + os.sep + "train_set.csv"
Operate_data = Action_json_data(file_path=datapath, verbose=-1, seed=1)
Texts = Operate_data.get_data()["texts"].tolist()
Ylab = Operate_data.get_data()["labels"].tolist()
Types = Operate_data.get_data()["types"].tolist()

Labels = [int(i) for i in Ylab]
Labels = np.array(Labels)

feature_list = [extract_features(payload, keywords, keyword_weights) for payload in Texts]
feature_array = np.array(feature_list)

print("Partition the data....")
train_set_x, train_set_y, train_set_id, validation_set_x, validation_set_y, validation_set_id = partition_data(
    feature_array, Types, Labels, split_seed, usetestflag=True)

dtrain = xgb.DMatrix(train_set_x, label=train_set_y)
dval = xgb.DMatrix(validation_set_x, label=validation_set_y)

param = {
    'max_depth': 6,
    'eta': 0.3,
    'objective': 'multi:softprob',
    'num_class': len(set(Labels)),
    'eval_metric': 'mlogloss'
}
evallist = [(dtrain, 'train'), (dval, 'eval')]

num_round = epochs
bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=patience)

model_save_dir = os.path.join(model_save_path, dataset)
os.makedirs(model_save_dir, exist_ok=True)
bst.save_model(os.path.join(model_save_dir, 'xgboost_model.json'))