import pandas as pd

# 读取已保存的HPD.csv文件
csv_file = './PIK.csv'  # 请将此路径替换为您的实际文件路径
data = pd.read_csv(csv_file)

# 随机打乱数据集
shuffled_data = data.sample(frac=1, random_state=815).reset_index(drop=True)

# 查看打乱后的数据集大小
print("打乱后的数据集大小: ", len(shuffled_data))

# 保存打乱后的数据集
shuffled_data.to_csv('./shuffled_data.csv', index=False)

print("打乱后的数据集已保存为 shuffled_data.csv。")
