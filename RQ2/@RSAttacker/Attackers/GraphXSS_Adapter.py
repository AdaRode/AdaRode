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
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader as GeoDataLoader
# 禁止所有warning
warnings.filterwarnings("ignore")
import yaml
import pickle

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config("Config/adv_test.yaml")
device = torch.device(config['parameters']['device'])

input_dim = 100  # Word2Vec vector size
hidden_dim = 128  # GCN hidden layer size
num_classes = 3  # Number of classes
max_sequence_length = 1000

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

def build_graph_data(device, dataset):
    with open("./Model/{}/GraphXSS/graph.pickle".format(dataset),"rb") as file:
        data_list = pickle.load(file)
    # print(data_list)
    # np_array = np.array(data_list)
    # data_tensor = torch.tensor(np_array)
    # data_tensor = data_tensor.to(device)
    
    data_tensor = data_list
    return data_tensor

def add_self_loops_if_empty(edge_index, num_nodes, device):
    if edge_index.numel() == 0:
        edge_index = torch.arange(0, num_nodes, dtype=torch.long, device=device).unsqueeze(0).repeat(2, 1)
    return edge_index

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch):
        num_nodes = x.size(0)
        edge_index = add_self_loops_if_empty(edge_index, num_nodes, x.device)
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class GraphXSSAdapter:
    def __init__(self, model_name, dataset):
        self.device = torch.device(config['parameters']['device'])
        w2v_model = Word2Vec.load(f"/nvme2n1/YangYJworks/xjw/YYJex/AdRandSample/Model/{dataset}/GraphXSS/word2vec.model")
        model = GCN(input_dim, hidden_dim, num_classes).to(self.device)
        model.load_state_dict(torch.load(f"/nvme2n1/YangYJworks/xjw/YYJex/AdRandSample/Model/{dataset}/GraphXSS/B.pth", map_location=self.device))

        self.model = model
        self.w2v_model = w2v_model
        self.model_name = model_name
        self.dataset = dataset

    def get_pred(self, input):
        input_ = [input]

        embedded_texts = []
        for text in input_:
            words = list(text)
            embedded_seq = [self.w2v_model.wv[word] for word in words if word in self.w2v_model.wv]
            if len(embedded_seq) > 0:
                embedded_texts.append(np.array(embedded_seq))
            else:
                embedded_texts.append(np.zeros((1, self.w2v_model.vector_size)))

        graph_data = build_graph_data(self.device, self.dataset)
        test_loader = GeoDataLoader(graph_data, batch_size=32, shuffle=False)

        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                outputs = self.model(data.x, data.edge_index, data.batch)
                predictions = outputs.argmax(dim=1)

            return [predictions.item()]

    def get_prob(self, input):
        input_ = [input]

        embedded_texts = []
        for text in input_:
            words = list(text)
            embedded_seq = [self.w2v_model.wv[word] for word in words if word in self.w2v_model.wv]
            if len(embedded_seq) > 0:
                embedded_texts.append(np.array(embedded_seq))
            else:
                embedded_texts.append(np.zeros((1, self.w2v_model.vector_size)))

        graph_data = build_graph_data(self.device, self.dataset)
        test_loader = GeoDataLoader(graph_data, batch_size=32, shuffle=False)

        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                outputs = self.model(data.x, data.edge_index, data.batch)
                prob = F.softmax(outputs, dim=1)
                numpy_list = prob.cpu().numpy().flatten().tolist()

        return numpy_list

# xss = "d"
# model_name = "GraphXSS"
# dataset = "PIK"
# victim_model = GraphXSSAdapter(model_name=model_name, dataset=dataset)

# print(victim_model.get_prob(xss))
# print(victim_model.get_pred(xss))
