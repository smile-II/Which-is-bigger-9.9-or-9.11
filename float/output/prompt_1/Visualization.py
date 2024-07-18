import json
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-whitegrid')

# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def load_results(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {i + 1}: {line.strip()}")
                print(f"Error message: {e}")
    return data

def analyze_errors(data):
    errors = [entry for entry in data if entry['Label'] != entry['Prediction']]
    return errors

def visualize_errors(errors, output_file):
    fig, axes = plt.subplots(10, 1, figsize=(20, 18), dpi=300, sharex=True, sharey=True)

    for i in range(10):
        a_values = [float(entry['A']) - i for entry in errors if int(float(entry['A'])) == i]
        b_values = [float(entry['B']) - int(float(entry['B'])) for entry in errors if int(float(entry['A'])) == i]
        
        if len(a_values) > 0 and len(b_values) > 0:
            heatmap, xedges, yedges = np.histogram2d(a_values, b_values, bins=[np.arange(0.0, 1.0, 0.01), np.arange(0.0, 1.0, 0.01)])  # 精确设置bins的范围
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im = axes[i].imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', interpolation='nearest')

        axes[i].set_title(f'Integer part {i}', fontsize=14)
        axes[i].tick_params(axis='both', which='major', labelsize=12)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    fig.text(0.5, 0.04, 'Decimal part of A', ha='center', fontsize=16)
    fig.text(0.04, 0.5, 'Decimal part of B', va='center', rotation='vertical', fontsize=16)
    fig.suptitle('Errors in Comparison (heatmap)', fontsize=20)

    plt.savefig(output_file.replace('.png', '_heatmap.png'), bbox_inches='tight', pad_inches=0.1)
    # plt.show()



if __name__ == "__main__":
    file_path = 'float/output/prompt_1/0-9_results_t=1.25-0.json'
    output_dir = 'float/output/prompt_1/analysis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    data = load_results(file_path)
    errors = analyze_errors(data)
    output_file = os.path.join(output_dir, 'comparison_errors.png')
    visualize_errors(errors, output_file)
    print(f"总错误数: {len(errors)}")
    print(f"错误可视化图已保存到 {output_file}")
