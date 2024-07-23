import json
import pandas as pd
import matplotlib.pyplot as plt

# 读取过滤后的 JSONL 文件
input_file = 'float/output/data_visualization/filtered_errors.json'

data = []
with open(input_file, 'r', encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line.strip()))

# 将数据加载到 Pandas DataFrame 中
df = pd.DataFrame(data)

# 转换 A 和 B 列为浮点数
df['A'] = df['A'].astype(float)
df['B'] = df['B'].astype(float)

# 创建不同温度值的散点图
fig, ax = plt.subplots()
colors = {'0': 'r', '0.25': 'g', '0.5': 'b', '0.75': 'c', '1': 'm', '1.25': 'y'}

for temp in df['temperature'].unique():
    subset = df[df['temperature'] == temp]
    ax.scatter(subset['A'], subset['B'], label=f't={temp}', color=colors.get(temp, 'k'))

ax.legend()
ax.set_xlabel('A')
ax.set_ylabel('B')
ax.set_title('错误预测的A值和B值散点图（按温度）')

# 显示图表
# plt.show()
plt.savefig('float/output/data_visualization/error_count_plot.png')
# 创建不同 prompt 类别的散点图
fig, ax = plt.subplots()
colors = {'prompt_0': 'r', 'prompt_1': 'g', 'prompt_2': 'b', 'prompt_3': 'c', 'prompt_4': 'm'}

for prompt in df['prompt'].unique():
    subset = df[df['prompt'] == prompt]
    ax.scatter(subset['A'], subset['B'], label=f'{prompt}', color=colors.get(prompt, 'k'))

ax.legend()
ax.set_xlabel('A')
ax.set_ylabel('B')
ax.set_title('错误预测的A值和B值散点图（按 prompt 类别）')

# 显示图表
# plt.show()
plt.savefig('float/output/data_visualization/error_count_plot.png')