import torch.optim as optim
import torch
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel, T5Tokenizer, T5ForConditionalGeneration
from transformers import XLNetTokenizer, XLNetModel
from transformers import BertTokenizer, BertModel  # 导入BERT相关库
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import json
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score, roc_auc_score
import yaml
from sklearn.metrics import (classification_report, precision_recall_fscore_support, 
                             precision_score, recall_score, f1_score, accuracy_score, 
                             roc_auc_score, matthews_corrcoef)
Device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

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
    


class TestModule:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def test_with_progress(self, test_dataloader):
        self.model.eval()
        correct_predictions = 0
        predictions = []
        true_labels = []
        self.preds = []
        self.label = []

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Testing"):
                input_ids, attention_mask, labels = tuple(t.to(self.device) for t in batch)
                outputs = self.model(input_ids, attention_mask=attention_mask)
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
        # Calculate precision, recall, and F1-score
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

class TestConfigLoader:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get_model_and_tokenizer(self, model_name='XLnet'):
        device = Device
        model_name = self.config[model_name]['name']
        num_labels = self.config[model_name]['num_labels']
        model_path = self.config[model_name]['path']
        tokenizer_path = self.config[model_name]['path']
        check_point_path = self.config[model_name]['checkpoint']
        
        if model_name == 'XLnet':
            tokenizer = XLNetTokenizer.from_pretrained(tokenizer_path)
            model = XLnetDet(model_path, num_labels).to(device)
            model.load_state_dict(torch.load(check_point_path, map_location=device))
        elif model_name == 'RoBERTa':
            tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
            model = RoBERTaDet(model_path, num_labels).to(device)
            model.load_state_dict(torch.load(check_point_path, map_location=device))
        elif model_name == 'T5':
            tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
            model = T5Det(model_path, num_labels).to(device)
            model.load_state_dict(torch.load(check_point_path, map_location=device))
        elif model_name == 'BERT':  # 添加BERT的处理逻辑
            tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            model = BERTDet(model_path, num_labels).to(device)
            model.load_state_dict(torch.load(check_point_path, map_location=device))
        return model, tokenizer

def tokenize_texts(texts, tokenizer, max_length=128):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt").to(device)

# Load configuration from YAML file
# 'BERT' 'RoBERTa' 'XLnet' 'T5' 
model_selection = 'RoBERTa'  # Change this to 'BERT' to test with BERT model
config_loader = TestConfigLoader('./Config/Test.yaml')
model, tokenizer = config_loader.get_model_and_tokenizer(model_name=model_selection)

# Load test data and set up test dataloader
import pickle
datasets = "./Data/HPD/test.pickle"
with open(datasets,"rb") as file:
    data = pickle.load(file)
# data = pd.read_csv(datasets)
print(data)

texts_test = data['adv_raw']
labels_test = data['adv_label']
# texts_test = data['ori_raw']
# labels_test = data['adv_label']
device = Device

test_inputs = tokenize_texts(texts_test, tokenizer)
test_dataset = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], torch.tensor(labels_test).to(device))
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize TestModule with the model and device
test_module = TestModule(model, device)
# Run the test with progress bar
test_module.test_with_progress(test_dataloader)
