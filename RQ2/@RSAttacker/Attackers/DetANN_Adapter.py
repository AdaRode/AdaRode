import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import XLNetTokenizer, XLNetModel
from urllib.parse import quote
import re
import random
import os
import numpy as np
import xgboost as xgb
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from joblib import load
import warnings
# 禁止所有warning
warnings.filterwarnings("ignore")
import yaml
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config("Config/adv_test.yaml")

device = torch.device(config['parameters']['device'])


keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'UNION', 'OR', 'AND', 'FROM', 'WHERE']
keyword_weights = {k: 1 for k in keywords}

# 定义特征提取函数
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



class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class BiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # Bidirectional, so hidden_dim * 2
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_out_mean = gru_out.mean(dim=1)  # Average pooling
        out = self.fc(gru_out_mean)
        return out

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # 双向LSTM * 2

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out_mean = lstm_out.mean(dim=1)  # 平均池化
        out = self.fc(lstm_out_mean)
        return out
    
    

class DNNAdapter:
    def __init__(self,model_name,dataset):
        device = torch.device(config['parameters']['device'])  # 将模型设置在第5个GPU上
        if model_name.endswith('DBN'):
            model = DNN(8,128,num_classes=3).to(device) 
            model.load_state_dict(torch.load(f"/nvme2n1/YangYJworks/xjw/YYJex/AdRandSample/Model/"+dataset+"/DetANN(DBN)/B.pth"))
        if model_name.endswith('BiGRU'):
            model = BiGRU(8,128,num_classes=3).to(device)
            model.load_state_dict(torch.load(f"/nvme2n1/YangYJworks/xjw/YYJex/AdRandSample/Model/"+dataset+"/DetANN(BiGRU)/B.pth", map_location=device))
        if model_name.endswith('BiLSTM'):
            model = BiLSTM(8,128,num_classes=3).to(device)
            model.load_state_dict(torch.load(f"/nvme2n1/YangYJworks/xjw/YYJex/AdRandSample/Model/"+dataset+"/DetANN(BiLSTM)/B.pth", map_location=device))
        if model_name.endswith('RF'):
            model = load(f"/nvme2n1/YangYJworks/xjw/YYJex/AdRandSample/Model/"+dataset+"/DetANN(RF)/random_forest_model.joblib")
        if model_name.endswith('XGB'):
            model = xgb.Booster()
            model.load_model("/nvme2n1/YangYJworks/xjw/YYJex/AdRandSample/Model/"+dataset+"/DetANN(XGB)/xgboost_model.json")
        self.model = model
        self.model_name = model_name

    def get_pred(self, input):
        input_=[]
        input_.append(input)
        test_feature_list = [extract_features(payload, keywords, keyword_weights) for payload in input_]
        test_feature_array = np.array(test_feature_list)
        if self.model_name.endswith('RF'):
            predictions = self.model.predict(test_feature_array)[0]
            return [predictions]
        if self.model_name.endswith('XGB'):
            dtest = xgb.DMatrix(test_feature_array)
            predictions = np.asarray([np.argmax(line) for line in self.model.predict(dtest)])
            return [predictions.item()]
        if self.model_name.endswith('DBN'):
            test_set_x = torch.tensor(test_feature_array, dtype=torch.float32)
        if self.model_name.endswith('BiGRU') or self.model_name.endswith('BiLSTM'):
            test_set_x = torch.tensor(test_feature_array, dtype=torch.float32).unsqueeze(1)  # 添加维度以符合 LSTM 输入格式
        test_dataset = TensorDataset(test_set_x)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 用于获取预测概率
        with torch.no_grad():
            if self.model_name.endswith('DBN') or self.model_name.endswith('BiGRU') or self.model_name.endswith('BiLSTM'):
                for batch in test_loader:
                    input_ids = batch[0]
                    input_ids = input_ids.to(device)
                    outputs = self.model(input_ids)
                    predictions = outputs.argmax(dim=1)


            return [predictions.item()]

    def get_prob(self, input):
        input_=[]
        input_.append(input)
        test_feature_list = [extract_features(payload, keywords, keyword_weights) for payload in input_]
        test_feature_array = np.array(test_feature_list)
        if self.model_name.endswith('RF'):
            prob = self.model.predict_proba(test_feature_array)[0].tolist()
            return prob
        if self.model_name.endswith('XGB'):
            dtest = xgb.DMatrix(test_feature_array)
            prob = self.model.predict(dtest)[0].tolist()
            return prob
        if self.model_name.endswith('DBN'):
            test_set_x = torch.tensor(test_feature_array, dtype=torch.float32)
        if self.model_name.endswith('BiGRU') or self.model_name.endswith('BiLSTM'):
            test_set_x = torch.tensor(test_feature_array, dtype=torch.float32).unsqueeze(1)  # 添加维度以符合 LSTM 输入格式
        test_dataset = TensorDataset(test_set_x)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        

        # 用于获取预测概率
        with torch.no_grad():
            if self.model_name.endswith('DBN') or self.model_name.endswith('BiGRU') or self.model_name.endswith('BiLSTM'):
                for batch in test_loader:
                    input_ids = batch[0]
                    input_ids = input_ids.to(device)
                    outputs = self.model(input_ids)
                    prob = F.softmax(outputs, dim=1)
                    numpy_list = prob.cpu().numpy().flatten().tolist()
        return numpy_list
    
# xss = "<script>alert(1)<script>"
# model_name="DetANN-DBN"
# # model_name="DetANN-BiGRU"
# # model_name="DetANN-BiLSTM"
# # model_name="DetANN-RF"
# # model_name="DetANN-XGB"
# dataset="PIK"
# victim_model = DNNAdapter(model_name=model_name,dataset=dataset)
# print(victim_model.get_prob(xss))
# print(victim_model.get_pred(xss))




