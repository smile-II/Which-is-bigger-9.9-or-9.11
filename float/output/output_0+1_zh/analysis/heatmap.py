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
    return errors

def visualize_errors(errors, output_file):
    a_values = [float(entry['A']) for entry in errors]
    b_values = [float(entry['B']) for entry in errors]

    # 使用热力图
    plt.figure(figsize=(20, 18), dpi=300)
    heatmap, xedges, yedges = np.histogram2d(a_values, b_values, bins=[np.arange(1.0, 10.0, 0.1), np.arange(1.0, 10.0, 0.1)])  # 精确设置bins的范围
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', interpolation='nearest')
    
    cbar = plt.colorbar( pad=0.05, fraction=0.045)  # 设置图例在底部，调整大小
    cbar.ax.tick_params(labelsize=20)  # 设置图例刻度文字大小
    # cbar = plt.colorbar(orientation='horizontal', pad=0.2, fraction=0.05)  # 设置图例在底部，调整大小

    plt.xlabel('Value A', fontsize=30)
    plt.ylabel('Value B', fontsize=30)
    plt.title('Errors in Comparison (heatmap)', fontsize=36)

    # 设置刻度
    plt.xticks(np.arange(1.0, 10.0, 0.1), fontsize=20)
    plt.yticks(np.arange(1.0, 10.0, 0.1), fontsize=20)

    # 设置网格线的样式和宽度
    plt.grid(which='both', color='gray', linestyle='-', linewidth=0.5)
    ax = plt.gca()

    # 设置次要网格线
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
    ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
    
    # 设置主要网格线
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax.grid(which='major', color='gray', linestyle='-', linewidth=1.0)

    # 控制图像边框的样式和宽度
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    plt.savefig(output_file.replace('.png', '_heatmap.png'), bbox_inches='tight', pad_inches=0.1)  # 使用 bbox_inches='tight' 确保图例大小和图片大小对齐
    # plt.show()

if __name__ == "__main__":
    file_path = 'float/output/1_results_t=1.25-0.json'
    output_dir = 'float/output/analysis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    data = load_results(file_path)
    errors = analyze_errors(data)
    output_file = os.path.join(output_dir, 'comparison_errors.png')
    visualize_errors(errors, output_file)
    print(f"总错误数: {len(errors)}")
    print(f"错误可视化图已保存到 {output_file}")
