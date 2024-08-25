import torch
import torch.nn as nn
import numpy as np
import os
import yaml
import pandas as pd
from gensim.models import Word2Vec
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import (classification_report, precision_recall_fscore_support, 
                             precision_score, recall_score, f1_score, accuracy_score, 
                             roc_auc_score, matthews_corrcoef)

# Define the padding function
def pad_sequences(sequences, maxlen, padding='post'):
    padded_sequences = np.zeros((len(sequences), maxlen, sequences[0].shape[1]))
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

# 数据读取模板类
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
    # 初始化函数
    def __init__(self, file_path, verbose=1, seed=1):
        super().__init__(file_path=file_path, verbose=verbose, seed=seed)

    def _read_data(self):
        if self.verbose > 0:
            print("Reading data from file:", self.file_path)
        data = pd.read_csv(self.file_path, encoding="utf-8")
        return data

    def get_data(self) -> dict:
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
class WordEmbeddingBuilder:
    def __init__(self, vector_size=100, window=5, min_count=1, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None
    
    def build_word2vec(self, sentences):
        print(f"Number of texts used for Word2Vec training: {len(sentences)}")
        print("Sample texts used for Word2Vec training:")
        for i in range(min(5, len(sentences))):
            print(' '.join(sentences[i]))
        
        self.model = Word2Vec(sentences, vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=self.workers)
        print(f"Number of unique words in Word2Vec vocabulary: {len(self.model.wv.index_to_key)}")
    
    def transform(self, texts):
        embedded_texts = []
        for text in tqdm(texts, desc="Transforming texts to embeddings"):
            words = text.split()
            embedded_seq = [self.model.wv[word] for word in words if word in self.model.wv]
            if len(embedded_seq) > 0:
                embedded_texts.append(np.array(embedded_seq))
            else:
                embedded_texts.append(np.zeros((1, self.vector_size)))
        return embedded_texts
    
    def save_model(self, path):
        self.model.save(path)
    
    def load_model(self, path):
        self.model = Word2Vec.load(path)

class TextCNN(nn.Module):
    def __init__(self, input_dim, num_classes, kernel_sizes=[3, 4, 5], num_filters=100):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, input_dim)) for k in kernel_sizes
        ])
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len, input_dim)
        convs = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        pools = [torch.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]
        out = torch.cat(pools, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

class TestModule:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def test_with_progress(self, test_dataloader):
        self.model.eval()
        correct_predictions = 0
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Testing"):
                input_ids, labels = tuple(t.to(self.device) for t in batch)
                outputs = self.model(input_ids)
                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == labels)
                predictions.extend(preds.view(-1).cpu().numpy())
                true_labels.extend(labels.view(-1).cpu().numpy())

        accuracy = correct_predictions.double() / len(test_dataloader.dataset)
        print(f'Test Accuracy: {accuracy:.4f}')

        self.label = np.array(true_labels)
        self.preds = np.array(predictions)
        self.test_multiple()
        self.test_xss()
        self.test_sql()

    def test_multiple(self):
        print("="*20+"Multi"+"="*20)
        y_true = self.label
        y_pred = self.preds
        report = classification_report(y_true, y_pred, digits=4)
        print(report)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-score: {f1:.4f}')
        
    def test_xss(self):
        print("="*20+"XSS"+"="*20)
        y_true = np.where(self.label == 2, 1, 0)
        y_pred = np.where(self.preds == 2, 1, 0)

        precision = precision_score(y_true, y_pred, pos_label=1)
        recall = recall_score(y_true, y_pred, pos_label=1)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        accuracy = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = float('nan')
        mcc = matthews_corrcoef(y_true, y_pred)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"MCC: {mcc:.4f}")

    def test_sql(self):
        print("="*20+"SQL"+"="*20)
        y_true = np.where(self.label == 1, 1, 0)
        y_pred = np.where(self.preds == 1, 1, 0)

        precision = precision_score(y_true, y_pred, pos_label=1)
        recall = recall_score(y_true, y_pred, pos_label=1)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        accuracy = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = float('nan')
        mcc = matthews_corrcoef(y_true, y_pred)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"MCC: {mcc:.4f}")

# 加载配置文件
config = yaml.safe_load(open("./TFModel/config/smallmodel.yaml", 'r', encoding="UTF-8"))

dataset = config.get('SmallModel').get('dataset')
embedding_method = config.get('SmallModel').get('embedding_method')
max_sequence_length = config.get('SmallModel').get('max_sequence_length')
model_save_path = config.get('SmallModel').get('model_save_path')

# 读取测试数据
# datapath = os.getcwd() + os.sep + "Data" + os.sep + dataset + os.sep + "test_set.csv"
# Operate_data = Action_json_data(file_path=datapath, verbose=-1, seed=1)
# Texts = Operate_data.get_data()["texts"].tolist()
# Ylab = Operate_data.get_data()["labels"].tolist()
# Types = Operate_data.get_data()["types"].tolist()

# Labels = [int(i) for i in Ylab]
# Labels = np.array(Labels)

import pickle
datasets = "./Data/PIK/test.pickle"
with open(datasets,"rb") as file:
    data = pickle.load(file)
print(data)
test_texts = data['adv_raw']
test_labels = data['adv_label']
# test_texts = data['ori_raw']
# test_labels = data['adv_label']
Texts = test_texts
Labels = test_labels
# 加载Word2Vec模型
w2v_model_save_path = os.path.join(model_save_path, dataset, "word2vec.model")
w2v_model = Word2Vec.load(w2v_model_save_path)

# 嵌入测试文本数据
embedded_texts = []
for text in Texts:
    words = text.split()
    embedded_seq = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    if len(embedded_seq) > 0:
        embedded_texts.append(np.array(embedded_seq))
    else:
        embedded_texts.append(np.zeros((1, w2v_model.vector_size)))  # 用零向量填充

padded_texts = pad_sequences(embedded_texts, max_sequence_length)

# 转换为 PyTorch tensors
test_set_x = torch.tensor(padded_texts, dtype=torch.float32)
test_set_y = torch.tensor(Labels, dtype=torch.long)

# 创建 Dataset 和 DataLoader
test_dataset = TensorDataset(test_set_x, test_set_y)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 打印一些信息
print(f"Test set size: {len(test_dataset)}")

# 检查是否有可用的 GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == 'cuda':
    gpu_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability(0)
    print(f"GPU Name: {gpu_name}")
    print(f"Compute Capability: {compute_capability[0]}.{compute_capability[1]}")

# 模型实例化
input_dim = 100  # Word2Vec vector size
num_classes = len(set(Labels))
hidden_dim = 128

model = TextCNN(input_dim=input_dim, num_classes=num_classes, kernel_sizes=[3, 4, 5], num_filters=100).to(device)



# 加载训练好的模型
best_model_path = os.path.join(model_save_path, dataset, 'B.pth')
model.load_state_dict(torch.load(best_model_path, map_location=device))

# 创建并评估模型
tester = TestModule(model, device)
tester.test_with_progress(test_loader)