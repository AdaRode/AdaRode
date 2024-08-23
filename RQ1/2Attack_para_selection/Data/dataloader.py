import pandas as pd
from sklearn.model_selection import train_test_split

# # 读取CSV文件
# df = pd.read_csv('./PIK/source.csv')  # 把'your_file.csv'替换成你的CSV文件路径


# # 添加新列，基于索引值进行判断
# df['type'] = df.index.to_series().apply(lambda x: 'sql' if x < 19790 else 'xss')

# # 保存修改后的CSV文件
# df.to_csv('./PIK/PIK.csv', index=False)
# 分割数据集
df = pd.read_csv('./PIK/PIK.csv') 
# 这里的test_size=0.2表示测试集占20%，训练集占80%
train_set, test_set = train_test_split(df, test_size=0.1, random_state=815)

# 查看分割后的数据集
print("训练集的大小: ", len(train_set))
print("测试集的大小: ", len(test_set))
# 保存训练集
train_set.to_csv('./PIK/train_set.csv', index=False)

# 保存测试集
test_set.to_csv('./PIK/test_set.csv', index=False)
