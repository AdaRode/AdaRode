import pandas as pd
from sklearn.model_selection import train_test_split
# 读取CSV文件
csv_file = './HttpParamsDataset/payload_full.csv'  # 请将此路径替换为您的实际文件路径
data = pd.read_csv(csv_file, usecols=['payload', 'attack_type'])

# 将attack_type转换为Label列
def convert_label(attack_type):
    if attack_type == 'norm':
        return 0
    elif attack_type == 'sqli':
        return 1
    elif attack_type == 'xss':
        return 2
    else:
        return -1

data['Label'] = data['attack_type'].apply(convert_label)

# 定义SQL注入和XSS攻击的简单检测函数
def is_sql_injection(payload):
    sql_keywords = ['SELECT', 'UNION', 'INSERT', 'DELETE', 'UPDATE', 'DROP', '--', ';', '/*', '*/']
    for keyword in sql_keywords:
        if keyword.lower() in payload.lower():
            return True
    return False

def is_xss_attack(payload):
    xss_patterns = ['<script>', '</script>', 'javascript:', 'onerror=', 'onload=']
    for pattern in xss_patterns:
        if pattern.lower() in payload.lower():
            return True
    return False

# 添加一列type，初始值为'normal'
data['type'] = 'normal'

# 根据payload检测SQL注入和XSS攻击，并更新type列
data.loc[data['payload'].apply(is_sql_injection), 'type'] = 'sql'
data.loc[data['payload'].apply(is_xss_attack), 'type'] = 'xss'

# 删除Label列中值为-1的行
data = data[data['Label'] != -1]

# 重命名列名
data = data.rename(columns={'payload': 'Text'})

# 保存为新的CSV文件
output_csv_file = './HPD.csv'  # 请将此路径替换为您希望保存的文件路径
data.to_csv(output_csv_file, index=False)

print("Data saved to", output_csv_file)
df = pd.read_csv(output_csv_file) 
# 这里的test_size=0.2表示测试集占20%，训练集占80%
train_set, test_set = train_test_split(df, test_size=0.2, random_state=815)

# 查看分割后的数据集
print("训练集的大小: ", len(train_set))
print("测试集的大小: ", len(test_set))
# 保存训练集
train_set.to_csv('./train_set.csv', index=False)

# 保存测试集
test_set.to_csv('./test_set.csv', index=False)