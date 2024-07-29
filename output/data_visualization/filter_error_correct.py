import json
import os

# 定义基础目录和输出文件
base_dir = 'output/model_output'  # 请将此路径替换为实际的基础目录路径
output_file = 'output/data_visualization/WIL_hallucination_prompt.json'
correct_output_file = 'output/data_visualization/WIL_hallucination_free_prompt.json'

filtered_results = []
correct_results = []

# 遍历每个 prompt 目录
for prompt_dir in os.listdir(base_dir):
    prompt_path = os.path.join(base_dir, prompt_dir)
    if os.path.isdir(prompt_path) and prompt_dir.startswith('prompt_'):
        # 遍历目录中的每个 JSON 文件
        for json_file in os.listdir(prompt_path):
            if json_file.endswith('.json') and json_file.startswith('0-9_results_t='):
                file_path = os.path.join(prompt_path, json_file)
                with open(file_path, 'r', encoding='utf-8') as file:
                    # 读取文件中的 JSON 行
                    for line in file:
                        try:
                            data = json.loads(line.strip())
                            # 检查预测是否错误
                            if data["Label"] != data["Prediction"]:
                                # 添加 prompt 和温度信息
                                data["prompt"] = prompt_dir
                                data["temperature"] = json_file.split('=')[1].split('-')[0]
                                filtered_results.append(data)
                            else:
                                # 添加 prompt 和温度信息到正确结果
                                data["prompt"] = prompt_dir
                                data["temperature"] = json_file.split('=')[1].split('-')[0]
                                correct_results.append(data)
                        except json.JSONDecodeError as e:
                            print(f'JSON 解码错误: {e} 文件: {file_path}')

# 将过滤后的错误结果保存到输出文件
with open(output_file, 'w', encoding='utf-8') as file:
    for result in filtered_results:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")

print(f'过滤后的错误结果已保存到 {output_file}')

# 将正确的结果保存到另一个输出文件
with open(correct_output_file, 'w', encoding='utf-8') as file:
    for result in correct_results:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")

print(f'正确的结果已保存到 {correct_output_file}')
