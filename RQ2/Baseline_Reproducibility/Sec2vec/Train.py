import yaml
import torch
import pandas as pd
import numpy as np
import random
import os
import time
import psutil
from transformers import RobertaTokenizer, RobertaModel
from transformers import XLNetTokenizer, XLNetModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ModelTrainer:
    def __init__(self, config_path):
        self.load_config(config_path)
        self.setup_environment()
        self.device = Device
        self.load_tokenizer_and_model()
        self.data_preparation()
        self.start_time = time.time()
        self.end_time = time.time()

    def get_runing_time(self):
        return self.end_time - self.start_time
    def load_tokenizer_and_model(self):
        if self.model_name == "RoBERTa":
            self.tokenizer = RobertaTokenizer.from_pretrained(self.pretrained_path)
            self.model = RoBERTaDet(self.pretrained_path, num_labels=self.num_labels).to(self.device)
        elif self.model_name == "XLnet":
            self.tokenizer = XLNetTokenizer.from_pretrained(self.pretrained_path)
            self.model = XLnetDet(self.pretrained_path, num_labels=self.num_labels).to(self.device)
        elif self.model_name == "T5":
            self.tokenizer = T5Tokenizer.from_pretrained(self.pretrained_path)
            self.model = T5Det(self.pretrained_path, num_labels=self.num_labels).to(self.device)
        elif self.model_name == "BERT":
            self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_path)
            self.model = BertDet(self.pretrained_path, num_labels=self.num_labels).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.dataset_path = os.path.join(os.getcwd(),"Data",config['Dataset'],"train_set.csv")
        self.model_name = config['Model']['Model_name']
        self.pretrained_path = config['Model']['PretrainedName']
        self.num_labels = config['Model']['NumLabels']
        self.epochs = config['Train']['Epoch']
        self.max_length = config['Train']['Max_length']
        self.lr = config['Train']['lr']
        self.batch_size = config['Train'].get('BatchSize', 32)
        self.dataset_name = config['Dataset']
        self.model_dir = os.path.join("./Model", self.dataset_name, self.model_name)
        os.makedirs(self.model_dir, exist_ok=True)
        print("=== Loaded Parameters ===")
        print(f"Dataset Path: {self.dataset_path}")
        print(f"Pretrained Model Name: {self.model_name}")
        print(f"Pretrained Model Path: {self.pretrained_path}")
        print(f"Number of Labels: {self.num_labels}")
        print(f"Epochs: {self.epochs}")
        print(f"Max Length: {self.max_length}")
        print(f"Learning Rate: {self.lr}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Dataset Name: {self.dataset_name}")
        print(f"Model Directory: {self.model_dir}")
        print("==========================")
        
    def setup_environment(self):
        seed_value = 42
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    
    def data_preparation(self):
        data = pd.read_csv(self.dataset_path)
        texts = data['Text'].tolist()
        labels = data['Label'].tolist()
        texts_train, texts_val, labels_train, labels_val = train_test_split(texts, labels, test_size=0.25, random_state=42)
        train_inputs = self.tokenize_texts(texts_train)
        val_inputs = self.tokenize_texts(texts_val)
        train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], torch.tensor(labels_train).to(self.device))
        val_dataset = TensorDataset(val_inputs['input_ids'], val_inputs['attention_mask'], torch.tensor(labels_val).to(self.device))
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def tokenize_texts(self, texts):
        return self.tokenizer(texts, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
    
    def train(self):
        best_val_loss = float('inf')
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1} / {self.epochs}')
            self.model.train()
            total_loss = 0
            for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}"):
                self.optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                loss, _ = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(self.train_dataloader)
            print(f'Epoch {epoch + 1}/{self.epochs}, Training Loss: {avg_train_loss:.4f}')
            val_loss = self.evaluate()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("Validation loss improved, saving model...")
                save_path = os.path.join(self.model_dir, f'model_epoch{epoch + 1}_val_loss{val_loss:.4f}.pth')
                torch.save(self.model.state_dict(), save_path)
                self.end_time = time.time()  
    
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                input_ids, attention_mask, labels = batch
                loss, _ = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_dataloader)
        print("Average Validation Loss:", avg_loss)
        return avg_loss

class RoBERTaDet(nn.Module):
    def __init__(self, model_name='roberta-base', num_labels=3):
        super(RoBERTaDet, self).__init__()
        self.num_labels = num_labels
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0, :])
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        return logits
    
class XLnetDet(nn.Module):
    def __init__(self, model_name='xlnet-base-cased', num_labels=3):  # 修改为XLNet相关的默认参数
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

class T5Det(nn.Module):
    def __init__(self, model_name='t5-based', num_labels=3):
        super(T5Det, self).__init__()
        self.num_labels = num_labels
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.pooler = nn.AdaptiveAvgPool1d(1)  # 添加池化层
        self.classifier = nn.Linear(self.t5.config.d_model, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask)  
        sequence_output = outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output.permute(0, 2, 1)).squeeze(-1)  
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        return logits

class BertDet(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=3):  
        super(BertDet, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0, :])
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        return logits

# 使用示例
if __name__ == "__main__":
    config_path = './Config/Train.yaml'  # 确保配置文件路径正确
    
    # Start the timer and memory usage tracking
    start_time = time.time()
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024 / 1024  # in GB
    
    trainer = ModelTrainer(config_path)
    trainer.train()
    
    # End the timer and memory usage tracking
    end_memory = process.memory_info().rss / 1024 / 1024 / 1024  # in GB
    
    total_time = trainer.get_runing_time()
    total_memory = end_memory - start_memory
    
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Total memory consumed: {total_memory:.2f} GB")
