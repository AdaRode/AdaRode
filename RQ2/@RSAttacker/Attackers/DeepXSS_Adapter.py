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
warnings.filterwarnings("ignore")
import yaml
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config("Config/adv_test.yaml")

device = torch.device(config['parameters']['device'])


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

class DBN(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes):
        super(DBN, self).__init__()
        self.rbm_layers = nn.ModuleList()
        self.hidden_dims = hidden_dims

        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                self.rbm_layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.rbm_layers.append(nn.Linear(hidden_dims[i-1], hidden_dim))

        self.fc = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        for rbm in self.rbm_layers:
            x = F.relu(rbm(x))
        out = self.fc(x)
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

class DeepXSSAdapter:
    def __init__(self,model_name,dataset):
        device = torch.device(config['parameters']['device'])  # 将模型设置在第5个GPU上
        if model_name.endswith('BiGRU'):
            w2v_model = Word2Vec.load(f"/nvme2n1/YangYJworks/xjw/YYJex/AdRandSample/Model/"+dataset+"/DeepXSS(BiGRU)/word2vec.model")
            model = BiGRUModel(input_dim, hidden_dim, num_layers, num_classes, dropout_prob).to(device)
            model.load_state_dict(torch.load(f"/nvme2n1/YangYJworks/xjw/YYJex/AdRandSample/Model/"+dataset+"/DeepXSS(BiGRU)/B.pth", map_location=device))        
        if model_name.endswith('BiLSTM'):
            w2v_model = Word2Vec.load(f"/nvme2n1/YangYJworks/xjw/YYJex/AdRandSample/Model/"+dataset+"/DeepXSS(BiLSTM)/word2vec.model")
            model = BiLSTMModel(input_dim, hidden_dim, num_layers, num_classes, dropout_prob).to(device)
            model.load_state_dict(torch.load(f"/nvme2n1/YangYJworks/xjw/YYJex/AdRandSample/Model/"+dataset+"/DeepXSS(BiLSTM)/B.pth", map_location=device))        
        if model_name.endswith('DBN'):
            w2v_model = Word2Vec.load(f"/nvme2n1/YangYJworks/xjw/YYJex/AdRandSample/Model/"+dataset+"/DeepXSS(DBN)/word2vec.model")
            model = DBN(100 * max_sequence_length, [512, 256], num_classes).to(device)
            model.load_state_dict(torch.load(f"/nvme2n1/YangYJworks/xjw/YYJex/AdRandSample/Model/"+dataset+"/DeepXSS(DBN)/B.pth", map_location=device))  
        if model_name.endswith('TR-IDS'):
            w2v_model = Word2Vec.load(f"/nvme2n1/YangYJworks/xjw/YYJex/AdRandSample/Model/"+dataset+"/TR-IDS/word2vec.model")
            model = TextCNN(input_dim=input_dim, num_classes=num_classes, kernel_sizes=[3, 4, 5], num_filters=100).to(device)    
            model.load_state_dict(torch.load(f"/nvme2n1/YangYJworks/xjw/YYJex/AdRandSample/Model/"+dataset+"/TR-IDS/B.pth", map_location=device))  
        if model_name.endswith('BLA'):
            w2v_model = Word2Vec.load(f"/nvme2n1/YangYJworks/xjw/YYJex/AdRandSample/Model/"+dataset+"/C-BLA/word2vec.model")
            model = C_BLA(input_dim, hidden_dim, num_classes).to(device)
            model.load_state_dict(torch.load(f"/nvme2n1/YangYJworks/xjw/YYJex/AdRandSample/Model/"+dataset+"/C-BLA/best_model.pth", map_location=device))   
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

# model_name="DeepXSS-BiGRU"
# # model_name="DeepXSS-BiLSTM"
# # model_name="DeepXSS-DBN"
# # model_name="TR-IDS"
# # model_name="C-BLA"
# dataset="HPD"
# victim_model = DeepXSSAdapter(model_name=model_name,dataset=dataset)

# print(victim_model.get_prob(xss))
# print(victim_model.get_pred(xss))




