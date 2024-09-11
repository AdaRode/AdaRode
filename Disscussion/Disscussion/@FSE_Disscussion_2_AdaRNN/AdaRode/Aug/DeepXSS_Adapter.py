import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import XLNetTokenizer, XLNetModel
from urllib.parse import quote
import re
import random
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from gensim.models import Word2Vec
from tqdm import tqdm
from joblib import load
import warnings
# 禁止所有warning
warnings.filterwarnings("ignore")
import yaml


device = torch.device("cuda:0") 


# Model instantiation
input_dim = 100  # Input dimension for each time step
hidden_dim = 128  # Hidden dimension for GRU
num_layers = 2  # Number of GRU layers
num_classes = 3
dropout_prob = 0.5  # Dropout probability
max_sequence_length = 1000


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


class BiGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout_prob=0.5):
        super(BiGRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bigru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 because of bidirectional
        self.dropout = nn.Dropout(dropout_prob)
        self.init_weights()

    def init_weights(self):
        for name, param in self.bigru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # *2 for bidirectional
        
        out, _ = self.bigru(x, h0)
        out = torch.mean(out, dim=1)  # Apply mean pooling
        out = self.fc(self.dropout(out))  # Pass through FC layer and dropout
        return out

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout_prob=0.5):
        super(BiLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 because of bidirectional
        self.dropout = nn.Dropout(dropout_prob)
        self.init_weights()

    def init_weights(self):
        for name, param in self.bilstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.bilstm(x, (h0, c0))
        out = torch.mean(out, dim=1)  # Apply mean pooling
        out = self.fc(self.dropout(out))  # Pass through FC layer and dropout
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout_prob=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out.permute(0, 2, 1)  # Change shape to (batch_size, hidden_dim, seq_len) for pooling
        out = self.avg_pool(out).squeeze(-1)  # Apply average pooling and remove last dimension
        out = self.dropout(torch.relu(self.fc1(out)))
        out = self.dropout(torch.relu(self.fc2(out)))
        out = self.fc3(out)
        return out




class DeepXSSAdapter:
    def __init__(self,model_name,dataset):      
        w2v_model = Word2Vec.load(f"/root/autodl-fs/@FSE_Disscussion_2_AdaRNN/AdaRode/Model/PIK/word2vec.model")
        model = BiLSTMModel(input_dim, hidden_dim, num_layers, num_classes, dropout_prob).to(device)
        model.load_state_dict(torch.load(f"/root/autodl-fs/@FSE_Disscussion_2_AdaRNN/AdaRode/Model/PIK/B.pth", map_location=device))         
        self.model = model
        self.w2v_model = w2v_model
        self.model_name = model_name
        

    def get_pred(self, input):
        # 用于获取预测结果
        input_=[]
        input_.append(input)
        
        # 嵌入测试文本数据
        embedded_texts = []
        for text in input_:
            words = text.split()
            # words = list(text)
            # print(words[0])
            embedded_seq = [self.w2v_model.wv[word] for word in words if word in self.w2v_model.wv]
            if len(embedded_seq) > 0:
                embedded_texts.append(np.array(embedded_seq))
            else:
                embedded_texts.append(np.zeros((1, self.w2v_model.vector_size)))  # 用零向量填充

        padded_texts = pad_sequences(embedded_texts, max_sequence_length)

        # 转换为 PyTorch tensors
        test_set_x = torch.tensor(padded_texts, dtype=torch.float32)
        # 创建 Dataset 和 DataLoader
        test_dataset = TensorDataset(test_set_x)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        # 用于获取预测概率
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch[0]
                input_ids = input_ids.to(device)
                outputs = self.model(input_ids)
                predictions = outputs.argmax(dim=1)


            return [predictions.item()]

    def get_prob(self, input):
        input_=[]
        input_.append(input)
        
        # 嵌入测试文本数据
        embedded_texts = []
        for text in input_:
            words = text.split()
            # words = list(text)
            # print(words[0])
            embedded_seq = [self.w2v_model.wv[word] for word in words if word in self.w2v_model.wv]
            if len(embedded_seq) > 0:
                embedded_texts.append(np.array(embedded_seq))
            else:
                embedded_texts.append(np.zeros((1, self.w2v_model.vector_size)))  # 用零向量填充

        padded_texts = pad_sequences(embedded_texts, max_sequence_length)

        # 转换为 PyTorch tensors
        test_set_x = torch.tensor(padded_texts, dtype=torch.float32)
        # 创建 Dataset 和 DataLoader
        test_dataset = TensorDataset(test_set_x)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 用于获取预测概率
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch[0]
                input_ids = input_ids.to(device)
                outputs = self.model(input_ids)
                prob = F.softmax(outputs, dim=1)
                numpy_list = prob.cpu().numpy().flatten().tolist()


        return numpy_list
    
# xss = "<crosssitescripting style=""crosssitescripting:expression(document.cookie=true)"">"


# # # model_name="DeepXSS-BiLSTM"
# model_name="DeepXSS-DBN"

# dataset="HPD"
# victim_model = DeepXSSAdapter(model_name="BiLSTM",dataset=dataset)

# print(victim_model.get_prob(xss))
# print(victim_model.get_pred(xss))




