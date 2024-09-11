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

class BiLSTM_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(BiLSTM_Attention, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        weighted_lstm_out = lstm_out * attention_weights
        return weighted_lstm_out.sum(dim=1), lstm_out

class ConvFeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super(ConvFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(input_dim, 128, kernel_size=4, padding=2)
        self.conv3 = nn.Conv1d(input_dim, 128, kernel_size=6, padding=3)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        out1 = torch.relu(self.conv1(x)).max(dim=2)[0]
        out2 = torch.relu(self.conv2(x)).max(dim=2)[0]
        out3 = torch.relu(self.conv3(x)).max(dim=2)[0]
        return torch.cat((out1, out2, out3), dim=1)

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.key = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.value = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.softmax(Q.bmm(K.transpose(1, 2)), dim=2)
        attention_out = attention_scores.bmm(V)

        pooled_out = self.pool(attention_out.transpose(1, 2)).squeeze(-1)
        return pooled_out

class C_BLA(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(C_BLA, self).__init__()
        self.bilstm_attention = BiLSTM_Attention(input_dim, hidden_dim, num_classes)
        self.conv_feature_extractor = ConvFeatureExtractor(input_dim)
        self.self_attention = SelfAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2 + 128 * 3 + hidden_dim * 2, num_classes)
        
    def forward(self, x):
        lstm_features, lstm_out = self.bilstm_attention(x)
        conv_features = self.conv_feature_extractor(x)
        attention_features = self.self_attention(lstm_out)
        combined_features = torch.cat((lstm_features, conv_features, attention_features), dim=1)
        out = self.fc(combined_features)
        return out

config = yaml.safe_load(open("./TFModel/config/smallmodel.yaml", 'r', encoding="UTF-8"))

dataset = config.get('SmallModel').get('dataset')
embedding_method = config.get('SmallModel').get('embedding_method')
max_sequence_length = config.get('SmallModel').get('max_sequence_length')
model_save_path = config.get('SmallModel').get('model_save_path')

import pickle
datasets = "./Data/{}/test.pickle".format(dataset)
with open(datasets,"rb") as file:
    data = pickle.load(file)
print(data)

# Random Attack data
# Texts = data['adv_raw']
# Labels = data['adv_label']

# Origin data for check model
Texts = data['ori_raw']
Labels = data['adv_label']
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


# load Word2Vec 模型
w2v_model_save_path = os.path.join(model_save_path, dataset, "word2vec.model")
w2v_model = Word2Vec.load(w2v_model_save_path)

# embedding text
embedded_texts = []
for text in Texts:
    words = text.split()
    embedded_seq = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    if len(embedded_seq) > 0:
        embedded_texts.append(np.array(embedded_seq))
    else:
        embedded_texts.append(np.zeros((1, w2v_model.vector_size)))  # 用零向量填充

padded_texts = pad_sequences(embedded_texts, max_sequence_length)

# Transform to PyTorch tensors
test_set_x = torch.tensor(padded_texts, dtype=torch.float32)
test_set_y = torch.tensor(Labels, dtype=torch.long)

# create Dataset and DataLoader
test_dataset = TensorDataset(test_set_x, test_set_y)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Test set size: {len(test_dataset)}")

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == 'cuda':
    gpu_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability(0)
    print(f"GPU Name: {gpu_name}")
    print(f"Compute Capability: {compute_capability[0]}.{compute_capability[1]}")

input_dim = 100  # Word2Vec vector size
num_classes = len(set(Labels))
hidden_dim = 128

model = C_BLA(input_dim, hidden_dim, num_classes).to(device)

best_model_path = os.path.join(model_save_path, dataset, 'best_model.pth')
model.load_state_dict(torch.load(best_model_path, map_location=device))

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
        # Generate the classification report
        report = classification_report(y_true, y_pred, digits=4)

        # Print the classification report
        print(report)
        # Calculate precision, recall, and F1-score
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-score: {f1:.4f}')
        
    def test_xss(self):
        print("="*20+"XSS"+"="*20)
        y_true = np.where(self.label == 2, 1, 0)
        y_pred = np.where(self.preds == 2, 1, 0)

        # Calculate metrics
        precision = precision_score(y_true, y_pred, pos_label=1)
        recall = recall_score(y_true, y_pred, pos_label=1)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        accuracy = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = float('nan')  # Handle cases with no positive (or negative) labels
        mcc = matthews_corrcoef(y_true, y_pred)
        # Print the metrics
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

        # Calculate metrics
        precision = precision_score(y_true, y_pred, pos_label=1)
        recall = recall_score(y_true, y_pred, pos_label=1)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        accuracy = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = float('nan')  # Handle cases with no positive (or negative) labels
        mcc = matthews_corrcoef(y_true, y_pred)

        # Print the metrics
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"MCC: {mcc:.4f}")

tester = TestModule(model, device)
tester.test_with_progress(test_loader)
