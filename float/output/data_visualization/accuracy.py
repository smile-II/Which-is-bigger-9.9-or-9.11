import os
import json
import matplotlib.pyplot as plt

# 定义文件目录
directories = ["prompt_0", "prompt_1", "prompt_2", "prompt_3", "prompt_4"]
file_pattern = "0-9_final_results_t={}-0.json"
temperatures = ["0.0", "0.25", "0.5", "0.75", "1.0", "1.25"]

# 存储结果的字典
accuracy_results = {}
error_counts = {}

# 读取文件并提取数据
for directory in directories:
    accuracy_results[directory] = []
    error_counts[directory] = []
    for threshold in temperatures:
        file_path = os.path.join(directory, file_pattern.format(threshold))
        print(f"正在处理文件: {file_path}")  # 调试输出文件路径
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    accuracy = data.get("accuracy", 0)
                    confusion_matrix = data.get("confusion_matrix", [[0, 0], [0, 0]])
                    error_count = confusion_matrix[0][1] + confusion_matrix[1][0]
                    
                    accuracy_results[directory].append(accuracy)
                    error_counts[directory].append(error_count)
                    print(f"文件已成功读取: {file_path}")  # 文件读取成功
            except Exception as e:
                print(f"读取文件时出错: {file_path}, 错误: {e}")
                accuracy_results[directory].append(None)
                error_counts[directory].append(None)
        else:
            print(f"文件不存在: {file_path}")  # 文件不存在
            accuracy_results[directory].append(None)
            error_counts[directory].append(None)

# 绘制正确率图
plt.figure(figsize=(12, 6))
for directory in directories:
    # 过滤掉空值
    accuracies = [acc for acc in accuracy_results[directory] if acc is not None]
    valid_temperatures = [temperatures[i] for i, acc in enumerate(accuracy_results[directory]) if acc is not None]
    if accuracies:  # 仅在有数据时绘制
        plt.plot(valid_temperatures, accuracies, marker='o', label=directory)
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.title('Accuracy for Different Prompts and Temperature')
plt.legend()
plt.grid(True)
plt.savefig('data_visualization/accuracy_plot.png')
# plt.show()

# 绘制错误数量图
plt.figure(figsize=(12, 6))
for directory in directories:
    # 过滤掉空值
    errors = [err for err in error_counts[directory] if err is not None]
    valid_temperatures = [temperatures[i] for i, err in enumerate(error_counts[directory]) if err is not None]
    if errors:  # 仅在有数据时绘制
        plt.plot(valid_temperatures, errors, marker='o', label=directory)
plt.xlabel('Threshold')
plt.ylabel('Error Count')
plt.title('Error Count for Different Prompts and Temperature')
plt.legend()
plt.grid(True)
plt.savefig('data_visualization/error_count_plot.png')
# plt.show()
