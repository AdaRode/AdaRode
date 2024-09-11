import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# 示例数据
parameters = [0.90, 0.93, 0.95, 0.97, 1.00]
overall_asr_counts = [64.84, 64.69, 65.11, 64.18, 64.02]

# 假设一个误差范围
error = [0.3, 0.4, 0.5, 0.45, 0.4]

# 插值
interp_parameters = np.linspace(min(parameters), max(parameters), 200)
interp_overall_asr_counts = np.interp(interp_parameters, parameters, overall_asr_counts)
interp_error = np.interp(interp_parameters, parameters, error)

# 调整图形的纵横比
plt.figure(figsize=(14, 3))  # 宽度14，高度5

# 绘制折线图和阴影区域
plt.plot(interp_parameters, interp_overall_asr_counts, label='ASR Change Curves', color='black')
plt.fill_between(interp_parameters, interp_overall_asr_counts - interp_error, interp_overall_asr_counts + interp_error, color='#0052c4', alpha=0.2)

# 绘制原始数据点
plt.scatter(parameters, overall_asr_counts, color='#0052c4', zorder=5)

# 只标注 (0.95, 65.11) 这个点
selected_x = 0.95
selected_y = 65.11
plt.scatter([selected_x], [selected_y], color='#bf0000', zorder=10)

# 添加半条垂直线和水平线
plt.plot([selected_x, selected_x], [61, selected_y], color='#bf0000', linestyle='--')
plt.plot([0.90, selected_x], [selected_y, selected_y], color='#bf0000', linestyle='--')

# 设置标签和y轴范围，并调整字体大小
plt.xlabel('Acceptance Rate', fontsize=28)
plt.ylabel('ASR (%)', fontsize=28)
plt.ylim(61.0, 66.5)  # 设置 y 轴范围
plt.xlim(0.905, 1.00)  # 设置 x 轴范围从 0.905 开始
plt.grid(True)

# 设置图例，字体大小为28，位置为右下角
plt.legend(fontsize=28, loc='lower right')

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

# 设置Y轴刻度为整数
formatter = FuncFormatter(lambda y, _: '{:.0f}'.format(y))
plt.gca().yaxis.set_major_formatter(formatter)

# 保存图片
plt.savefig('output_plot.png', bbox_inches='tight')
plt.show()
