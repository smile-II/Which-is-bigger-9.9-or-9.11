import json
import pandas as pd
import matplotlib.pyplot as plt

# 读取过滤后的 JSON 文件
input_file = 'float/output/data_visualization/filtered_errors.json'

data = []
with open(input_file, 'r', encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line.strip()))

# 将数据加载到 Pandas DataFrame 中
df = pd.DataFrame(data)

# 统计每个 B 选项和 prompt 组合的错误数量
error_counts = df.groupby(['B', 'prompt']).size().unstack(fill_value=0)

# 计算每个 B 选项的总错误数量并排序
total_errors = error_counts.sum(axis=1).sort_values(ascending=False)
error_counts = error_counts.loc[total_errors.index]

# 创建颜色映射
colors = {'prompt_0': 'skyblue', 'prompt_1': 'orange', 'prompt_2': 'green', 'prompt_3': 'red', 'prompt_4': 'purple'}

# 创建堆叠柱状图
fig, ax = plt.subplots(figsize=(12, 6))

error_counts.plot(kind='bar', stacked=True, ax=ax, color=[colors[prompt] for prompt in error_counts.columns])

ax.set_xlabel('B Option')
ax.set_ylabel('Number of Errors')
ax.set_title('Number of Errors for Each B Option by Prompt')

# 旋转横坐标上的数字
plt.xticks(rotation=90, fontsize=8)

# 添加图例
ax.legend(title='Prompt')

# 保存图表
# plt.savefig('float/output/data_visualization/error_count_by_B_and_prompt.png')
plt.savefig('float/output/data_visualization/error_count_by_B_and_prompt.png', dpi=300)
# 显示图表
# plt.show()
