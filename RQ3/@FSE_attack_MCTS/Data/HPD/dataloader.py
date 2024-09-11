import pandas as pd

# 读取已保存的HPD.csv文件
csv_file = './HPD.csv'  # 请将此路径替换为您的实际文件路径
data = pd.read_csv(csv_file)

# 随机打乱数据集
shuffled_data = data.sample(frac=1, random_state=815).reset_index(drop=True)

# 查看打乱后的数据集大小
print("打乱后的数据集大小: ", len(shuffled_data))

# 计算每份数据集的大小
num_splits = 10
split_size = len(shuffled_data) // num_splits

# 分割数据集并保存每一份
for i in range(num_splits):
    start_idx = i * split_size
    end_idx = (i + 1) * split_size if i != num_splits - 1 else len(shuffled_data)
    split_data = shuffled_data.iloc[start_idx:end_idx]
    split_filename = f'./shuffled_data_part_{i+1}.csv'
    split_data.to_csv(split_filename, index=False)
    print(f"打乱后的数据集第 {i+1} 份已保存为 {split_filename}。")
