import json
import numpy as np
import matplotlib.pyplot as plt

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

def visualize_data(data, output_file):
    fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(30, 5), sharey=True)
    
    for i in range(10):
        ax = axes[i]
        ax.set_title(f'Integer part: {i}')
        subset = [entry for entry in data if entry['value of the integer part'] == i]
        
        if subset:
            a_values = [float(entry['A']) - i for entry in subset]  # 减去整数部分的值
            b_values = [float(entry['B']) - i for entry in subset]  # 减去整数部分的值
            correct_predictions = [entry for entry in subset if entry['Label'] == entry['Prediction']]
            incorrect_predictions = [entry for entry in subset if entry['Label'] != entry['Prediction']]
            
            # ax.scatter([float(entry['A']) - i for entry in correct_predictions], 
            #            [float(entry['B']) - i for entry in correct_predictions], 
            #            c='blue', label='Correct', s=10, alpha=0.1)
            ax.scatter([float(entry['A']) - i for entry in incorrect_predictions], 
                       [float(entry['B']) - i for entry in incorrect_predictions], 
                       c='red', label='Incorrect', s=10)
        # axes[i].set_yticks(np.arange(0.0, 0.2, 0.01))
        # axes[i].set_xticks(np.arange(0.0, 1.0, 0.1))
        # axes[i].set_ylim(0.0, 0.2)
        # axes[i].set_xlim(0.0, 1.0)
        # axes[i].set_yticks(np.arange(0.0, 0.5, 0.01))
        # axes[i].set_xticks(np.arange(0.0, 1.0, 0.1))
        # axes[i].set_ylim(0.0, 0.5)
        # axes[i].set_xlim(0.0, 1.0)
        
        ax.set_xlabel('Value A - Decimal Part')
        if i == 0:
            ax.set_ylabel('Value B - Decimal Part')
        
        if i == 9:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file)
    # plt.show()

if __name__ == "__main__":
    # file_path = 'float/output/prompt_1/0-9_results_t=1.25-0.json'  # 确保路径正确
    file_path = r'float\output\prompt_0\all_results_t=1.0-0.json'  # 确保路径正确
    
    output_file = 'float/output/prompt_0/all_visualization-0.png'
    
    data = load_results(file_path)
    visualize_data(data, output_file)
    print(f"Visualization saved to {output_file}")
