import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os
import yaml
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, matthews_corrcoef

# 读取配置文件
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

# 数据读取模板类
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

# 关键词和权重
keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'UNION', 'OR', 'AND', 'FROM', 'WHERE']
keyword_weights = {k: 1 for k in keywords}

# 模型定义
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


# 加载测试数据
import pickle
datasets = "./Data/HPD/test.pickle"
with open(datasets,"rb") as file:
    data = pickle.load(file)
print(data)
test_texts = data['adv_raw']
test_labels = data['adv_label']
# test_texts = data['ori_raw']
# test_labels = data['adv_label']
Texts = test_texts
Labels = np.array(test_labels)

# 提取特征
feature_list = [extract_features(payload, keywords, keyword_weights) for payload in Texts]
feature_array = np.array(feature_list)

# 转换为 PyTorch tensors
test_set_x = torch.tensor(feature_array, dtype=torch.float32).unsqueeze(1)  # 添加维度以符合 LSTM 输入格式
test_set_y = torch.tensor(Labels, dtype=torch.long)

# 创建 Dataset 和 DataLoader
test_dataset = TensorDataset(test_set_x, test_set_y)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 检查是否有可用的 GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载模型
input_dim = feature_array.shape[1]
num_classes = len(set(Labels))
model = BiGRU(input_dim, 128, num_classes).to(device)

# 加载训练好的模型权重
model_path = os.path.join(model_save_path, dataset, 'B.pth')
model.load_state_dict(torch.load(model_path, map_location=device))

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 测试模型并计算指标
def test(model, test_loader, cmax_sequence_lengthriterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids, labels = batch
            input_ids, labels = input_ids.to(device), labels.to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_test_loss = total_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.4f}')

    # 存储标签和预测
    label = np.array(true_labels)
    preds = np.array(predictions)
    
    # 计算和打印指标
    test_multiple(label, preds)
    test_xss(label, preds)
    test_sql(label, preds)

    return avg_test_loss, accuracy

def test_multiple(y_true, y_pred):
    print("="*20+"Multi"+"="*20)
    report = classification_report(y_true, y_pred, digits=4)
    print(report)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')

def test_xss(y_true, y_pred):
    print("="*20+"XSS"+"="*20)
    y_true = np.where(y_true == 2, 1, 0)
    y_pred = np.where(y_pred == 2, 1, 0)
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

def test_sql(y_true, y_pred):
    print("="*20+"SQL"+"="*20)
    y_true = np.where(y_true == 1, 1, 0)
    y_pred = np.where(y_pred == 1, 1, 0)
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

# 运行测试
test_loss, test_accuracy = test(model, test_loader, criterion, device)
