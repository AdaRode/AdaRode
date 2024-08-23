'''
Author: Yanjing Yang
Date: 2024-08-16 02:19:14
FilePath: \2Attack_para_selection\Result\Iter_analysis.py
Description: 

Copyright (c) 2024 by NJU(Nanjing University), All Rights Reserved. 
'''
import matplotlib.pyplot as plt
import numpy as np

# 示例数据
parameters = [10,50,100,150,200]
overall_asr_counts = [54.53, 63.40, 65.11, 64.94, 64.80]

# 假设一个误差范围
error = [1.0, 1.3, 1.5, 0.96, 0.83]

# 插值
interp_parameters = np.linspace(min(parameters), max(parameters), 200)
interp_overall_asr_counts = np.interp(interp_parameters, parameters, overall_asr_counts)
interp_error = np.interp(interp_parameters, parameters, error)

# 绘制折线图和阴影区域
plt.plot(interp_parameters, interp_overall_asr_counts, label='ASR Change Curves', color='black')
plt.fill_between(interp_parameters, interp_overall_asr_counts - interp_error, interp_overall_asr_counts + interp_error, color='#0052c4', alpha=0.2)

# 绘制原始数据点
plt.scatter(parameters, overall_asr_counts, color='#0052c4', zorder=5)

# 只标注 (0.95, 65.11) 这个点
selected_x = 100
selected_y = 65.11
plt.scatter([selected_x], [selected_y], color='#bf0000', zorder=10)

# 添加半条垂直线和水平线
plt.plot([selected_x, selected_x], [50, selected_y], color='#bf0000', linestyle='--')
plt.plot([0.90, selected_x], [selected_y, selected_y], color='#bf0000', linestyle='--')

# # 标注坐标轴上的值
# plt.text(selected_x-1, 60, f'{selected_x}', ha='center', va='bottom', fontsize=20, color='blue')
# plt.text(10+15, selected_y + 0.4, f'{selected_y}', ha='right', va='center', fontsize=20, color='blue')  # 右上移动一些

# 设置标签和y轴范围，并调整字体大小
plt.xlabel('Max Sampling Iteration', fontsize=24)
plt.ylabel('Overall ASR (%)', fontsize=24)
plt.ylim(50, 70)  # 设置 y 轴范围
plt.xlim(10 ,200)  # 设置 x 轴范围从 0.90 开始
plt.grid(True)
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# 保存图片
plt.savefig('output_plot_ITER.png', bbox_inches='tight')
plt.show()
