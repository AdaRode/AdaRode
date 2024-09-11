import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os
import yaml
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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

# Data reading template class
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
    # Initialization function
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

# Feature extraction function
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

# Keywords and weights
keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'UNION', 'OR', 'AND', 'FROM', 'WHERE']
keyword_weights = {k: 1 for k in keywords}

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

# Data partition function
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

# Training and validation function
class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, epochs, model_dir, device, dataset, patience=5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.model_dir = model_dir
        self.device = device
        self.dataset = dataset
        self.patience = patience
        self.no_improve_count = 0

    def train(self):
        best_val_loss = float('inf')
        os.makedirs(os.path.join(self.model_dir, self.dataset), exist_ok=True)
        
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1} / {self.epochs}')
            self.model.train()
            total_loss = 0
            for batch in tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}"):
                input_ids, labels = batch
                input_ids, labels = input_ids.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(input_ids)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(self.train_loader)
            print(f'Epoch {epoch + 1}/{self.epochs}, Training Loss: {avg_train_loss:.4f}')
            
            val_loss = self.evaluate()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model.state_dict()
                self.no_improve_count = 0
                print("Validation loss improved, saving model...")
                save_path = os.path.join(self.model_dir, self.dataset, f'model_epoch{epoch + 1}_val_loss{val_loss:.4f}.pth')
                torch.save(self.model.state_dict(), save_path)
            else:
                self.no_improve_count += 1
                print(f"No improvement in validation loss for {self.no_improve_count} consecutive epochs.")
            
            if self.no_improve_count >= self.patience:
                print("Early stopping triggered. Training halted.")
                save_path = os.path.join(self.model_dir, self.dataset, f'best_model.pth')
                torch.save(best_model, save_path)
                break

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids, labels = batch
                input_ids, labels = input_ids.to(self.device), labels.to(self.device)
                outputs = self.model(input_ids)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        avg_val_loss = total_loss / len(self.val_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}')
        return avg_val_loss

# Load configuration file
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

# Read data
datapath = os.getcwd() + os.sep + "Data" + os.sep + dataset + os.sep + "train_set.csv"
Operate_data = Action_json_data(file_path=datapath, verbose=-1, seed=1)
Texts = Operate_data.get_data()["texts"].tolist()
Ylab = Operate_data.get_data()["labels"].tolist()
Types = Operate_data.get_data()["types"].tolist()

Labels = [int(i) for i in Ylab]
Labels = np.array(Labels)

# Extract features
feature_list = [extract_features(payload, keywords, keyword_weights) for payload in Texts]
feature_array = np.array(feature_list)

# Data partition
print("Partitioning the data....")
train_set_x, train_set_y, train_set_id, validation_set_x, validation_set_y, validation_set_id = partition_data(
    feature_array, Types, Labels, split_seed, usetestflag=True)

# Convert to PyTorch tensors
train_set_x = torch.tensor(train_set_x, dtype=torch.float32)  # No need to unsqueeze for DNN
train_set_y = torch.tensor(train_set_y, dtype=torch.long)
validation_set_x = torch.tensor(validation_set_x, dtype=torch.float32)  # No need to unsqueeze for DNN
validation_set_y = torch.tensor(validation_set_y, dtype=torch.long)

# Create Dataset and DataLoader
train_dataset = TensorDataset(train_set_x, train_set_y)
val_dataset = TensorDataset(validation_set_x, validation_set_y)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Print some information
print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == 'cuda':
    gpu_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability(0)
    print(f"GPU Name: {gpu_name}")
    print(f"Compute Capability: {compute_capability[0]}.{compute_capability[1]}")

# Model instantiation
input_dim = feature_array.shape[1]
num_classes = len(set(Labels))
hidden_dim = 128 
model = DNN(input_dim, hidden_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create and train model
trainer = ModelTrainer(model, train_loader, val_loader, criterion,
                        optimizer, epochs, model_save_path, 
                        device, dataset, patience)
trainer.train()
