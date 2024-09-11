import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os
import yaml
import pickle
from tqdm import tqdm
from gensim.models import Word2Vec

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
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient Clipping
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
                save_path = os.path.join(self.model_dir, self.dataset, f'B.pth')
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

# 加载配置文件
config = yaml.safe_load(open("./TFModel/config/smallmodel.yaml", 'r', encoding="UTF-8"))

dataset = config.get('SmallModel').get('dataset')
max_sequence_length = config.get('SmallModel').get('max_sequence_length')
split_seed = config.get('SmallModel').get('split_seed')
model_save_path = config.get('SmallModel').get('model_save_path')
epochs = config.get('SmallModel').get('epochs')
patience = config.get('SmallModel').get('patience')

# 加载增强后的数据
with open("./Data/AugmentedData.pickle", "rb") as file:
    augmented_data = pickle.load(file)

aug_x = augmented_data["adv_raw"]
aug_y = augmented_data["adv_label"]

# 加载预训练的Word2Vec模型
w2v_model = Word2Vec.load(f"/root/autodl-fs/@FSE_Disscussion_2_AdaRNN/AdaRode/Model/PIK/word2vec.model")

# 转换增强后的数据为词向量
def transform(texts, w2v_model, vector_size):
    embedded_texts = []
    for text in tqdm(texts, desc="Transforming texts to embeddings"):
        words = text.split()
        embedded_seq = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
        if len(embedded_seq) > 0:
            embedded_texts.append(np.array(embedded_seq))
        else:
            embedded_texts.append(np.zeros((1, vector_size)))  # 用零向量填充
    return embedded_texts

aug_x_vectors = transform(aug_x, w2v_model, w2v_model.vector_size)
aug_x_padded = pad_sequences(aug_x_vectors, max_sequence_length)

# 转换为 PyTorch tensors
aug_x_padded = torch.tensor(aug_x_padded, dtype=torch.float32)
aug_y_tensors = torch.tensor(aug_y, dtype=torch.long)

# 创建增强数据的 Dataset 和 DataLoader
aug_dataset = TensorDataset(aug_x_padded, aug_y_tensors)
aug_loader = DataLoader(aug_dataset, batch_size=32, shuffle=True)

# 分割验证集
train_size = int(0.8 * len(aug_dataset))
val_size = len(aug_dataset) - train_size
train_dataset, val_dataset = random_split(aug_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 打印一些信息
print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == 'cuda':
    gpu_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability(0)
    print(f"GPU Name: {gpu_name}")
    print(f"Compute Capability: {compute_capability[0]}.{compute_capability[1]}")

# 模型实例化
input_dim = w2v_model.vector_size  # Input dimension is the size of word vectors
hidden_dim = 128  # Hidden dimension for LSTM layers
num_layers = 2  # Number of LSTM layers
num_classes = 3  # 设置为三分类

model = BiLSTMModel(input_dim, hidden_dim, num_layers, num_classes, dropout_prob=0.5).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 加载预训练的模型参数
pretrained_model_path = os.path.join("/root/autodl-fs/@FSE_Disscussion_2_AdaRNN/AdaRode/Model/PIK/B.pth")
pretrained_state_dict = torch.load(pretrained_model_path, map_location=device)

# Modify the state dictionary to match the new model's dimensions if necessary
# pretrained_state_dict['fc.weight'] = torch.nn.init.xavier_uniform_(torch.empty(num_classes, hidden_dim * 2))
# pretrained_state_dict['fc.bias'] = torch.nn.init.zeros_(torch.empty(num_classes))

# Load the modified state dictionary into the new model
model.load_state_dict(pretrained_state_dict, strict=False)

# 创建并训练模型
trainer = ModelTrainer(model, train_loader, val_loader, criterion,
                        optimizer, epochs, model_save_path, 
                        device, dataset, patience)
trainer.train()
