import json
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-whitegrid')
# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False     # 正常显示负号

def load_results(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def analyze_errors(data):
    errors = [entry for entry in data if entry['标签'] != entry['预测']]
    corrects = [entry for entry in data if entry['标签'] == entry['预测']]
    return errors, corrects

def visualize_errors(errors, corrects, output_file):
    a_values_errors = [float(entry['A']) for entry in errors]
    b_decimal_parts_errors = [float(entry['B']) % 1 for entry in errors]
    b_values_adjusted_errors = [int(float(entry['A'])) + decimal_part for entry, decimal_part in zip(errors, b_decimal_parts_errors)]

    a_values_corrects = [float(entry['A']) for entry in corrects]
    b_decimal_parts_corrects = [float(entry['B']) % 1 for entry in corrects]
    b_values_adjusted_corrects = [int(float(entry['A'])) + decimal_part for entry, decimal_part in zip(corrects, b_decimal_parts_corrects)]

    plt.figure(figsize=(20, 18), dpi=100)
    plt.scatter(a_values_errors, b_values_adjusted_errors, c='red', alpha=0.5, label='Errors', s=10)  # 设置散点大小为10
    plt.scatter(a_values_corrects, b_values_adjusted_corrects, c='blue', alpha=0.5, label='Corrects', s=10)  # 设置散点大小为10

    plt.xlabel('Value A', fontsize=20)
    plt.ylabel('Integer Part of A + Decimal Part of B', fontsize=20)
    plt.title('Errors in Comparison', fontsize=24)
    plt.xticks(np.arange(1.1, 10.0, 0.5), fontsize=16)
    plt.yticks(np.arange(1.0, 10.0, 0.5), fontsize=16)

    plt.grid(True, linestyle='-', color='gray', linewidth=0.5)

    plt.legend()
    plt.savefig(output_file.replace('.png', '_scatter.png'))
    # plt.show()

if __name__ == "__main__":
    file_path = 'float/output/1_results_t=1.25-0.json'
    output_dir = 'float/output/analysis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    data = load_results(file_path)
    errors, corrects = analyze_errors(data)
    output_file = os.path.join(output_dir, 'comparison_errors.png')
    visualize_errors(errors, corrects, output_file)
    print(f"总错误数: {len(errors)}")
    print(f"错误可视化图已保存到 {output_file}")
