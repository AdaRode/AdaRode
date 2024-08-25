import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel, T5Tokenizer, T5ForConditionalGeneration
from transformers import XLNetTokenizer, XLNetModel
from transformers import RobertaTokenizer, RobertaModel, T5Tokenizer, T5ForConditionalGeneration
from transformers import BertTokenizer, BertModel  
from urllib.parse import quote
import re
import random
import os
import yaml
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config("Config/adv_test.yaml")

device = torch.device(config['parameters']['device'])

num_labels=3


class RoBERTaDet(nn.Module):
    def __init__(self, model_name='/nvme2n1/PTMs/roberta-base', num_labels=3):
        super(RoBERTaDet, self).__init__()
        self.num_labels = num_labels
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0, :])  # 取CLS token的输出进行分类
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        return logits

class XLnetDet(nn.Module):
    def __init__(self, model_name='/nvme2n1/YangYJworks/ADV/xlnet-base-cased', num_labels=3):  # 修改为XLNet相关的默认参数
        super(XLnetDet, self).__init__()
        self.num_labels = num_labels
        self.xlnet = XLNetModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.xlnet.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.xlnet(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0, :])
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        return logits
       

class BERTDet(nn.Module):  # 添加BERT相关的模型类
    def __init__(self, model_name, num_labels=3):
        super(BERTDet, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0, :])  # 取CLS token的输出进行分类
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        return logits


class Sec2vecAdapter:
    def __init__(self, model_name, dataset):
        device = torch.device(config['parameters']['device'])  # 将模型设置在第5个GPU上
        
        if model_name == 'RoBERTa':
            tokenizer = RobertaTokenizer.from_pretrained('/nvme2n1/PTMs/roberta-base')
            model = RoBERTaDet('/nvme2n1/PTMs/roberta-base', num_labels).to(device)
            model.load_state_dict(torch.load('/nvme2n1/YangYJworks/xjw/YYJex/AdRandSample/Model/'+dataset+'/Sec2Vec/best.pth', map_location=device))
        
        if model_name == 'XLNet':
            tokenizer = XLNetTokenizer.from_pretrained('/nvme2n1/YangYJworks/ADV/xlnet-base-cased')
            model = XLnetDet('/nvme2n1/YangYJworks/ADV/xlnet-base-cased', num_labels).to(device)
            model.load_state_dict(torch.load('/nvme2n1/YangYJworks/xjw/YYJex/AdRandSample/Model/'+dataset+'/AdaRodiaN/best.pth', map_location=device))
        
        if model_name == 'Ada':
            tokenizer = XLNetTokenizer.from_pretrained('/nvme2n1/YangYJworks/ADV/xlnet-base-cased')
            model = XLnetDet('/nvme2n1/YangYJworks/ADV/xlnet-base-cased', num_labels).to(device)
            model.load_state_dict(torch.load('/nvme2n1/YangYJworks/xjw/YYJex/AdRandSample/Model/'+dataset+'/AdaRodiaA/adaRodia.pth', map_location=device))

        self.model = model
        self.tokenizer = tokenizer

    def get_pred(self, input):
        # 用于获取预测结果
        res = self.tokenizer(input, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        input_ids, attention_mask = res["input_ids"], res["attention_mask"]
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            return [predictions.item()]

    def get_prob(self, input):
        res = self.tokenizer(input, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        input_ids, attention_mask = res["input_ids"], res["attention_mask"]
        input_ids.to(device)
        attention_mask.to(device)
        # 用于获取预测概率
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        prob = F.softmax(outputs, dim=1)
        numpy_list = prob.cpu().numpy().flatten().tolist()

        # Calculate the sum of the remaining numbers
        return numpy_list
    
# xss = "<crosssitescripting style=""crosssitescripting:expression(document.cookie=true)"">"

# # model_name="RoBERTa"
# model_name="XLNet"
# dataset="HPD"
# victim_model = Sec2vecAdapter(model_name=model_name,dataset=dataset)
# print(victim_model.get_prob(xss))
# print(victim_model.get_pred(xss))



