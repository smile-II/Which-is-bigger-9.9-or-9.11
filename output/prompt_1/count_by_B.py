import json
import pandas as pd
import matplotlib.pyplot as plt

# 读取过滤后的 JSON 文件
input_file = r'D:\project\WIL\output\prompt_1\greater_than_11_comparisons_10times_results_t=0.0-0.json'

# 加载 JSON 文件中的数据
data = []
with open(input_file, 'r', encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line.strip()))

# 将数据加载到 Pandas DataFrame 中
df = pd.DataFrame(data)
# print(df.head())
# 统计每个 B 选项的数量
# 过滤 Prediction 为 "B" 的数据
df_filtered = df[df['Prediction'] == 'B']
error_counts = df_filtered['B'].value_counts().sort_values(ascending=False)

# 统计每个 B 选项和 prompt 组合的错误数量
# error_counts = df.groupby(['B', 'prompt']).size().unstack(fill_value=0)

# 计算每个 B 选项的总错误数量并排序
# total_errors = error_counts.sum(axis=1).sort_values(ascending=False)
# error_counts = error_counts.loc[total_errors.index]

# 创建颜色映射（可选）
# colors = {'prompt_0': 'skyblue', 'prompt_1': 'orange', 'prompt_2': 'green', 'prompt_3': 'red', 'prompt_4': 'purple'}

# 创建堆叠柱状图
fig, ax = plt.subplots(figsize=(12, 6))

error_counts.plot(kind='bar', stacked=True, ax=ax)
# 如果要使用颜色映射，取消注释下面一行
# error_counts.plot(kind='bar', stacked=True, ax=ax, color=[colors[prompt] for prompt in error_counts.columns])

ax.set_xlabel('B Option')
ax.set_ylabel('Number of Errors')
ax.set_title('Number of Errors for Each B Option by Prompt')

# 旋转横坐标上的数字
plt.xticks(rotation=90, fontsize=8)

# 添加图例
ax.legend(title='Prompt')

# 保存图表
output_file = input_file.replace('.json', '.png')
plt.savefig(output_file, dpi=300)
# 显示图表
plt.show()
