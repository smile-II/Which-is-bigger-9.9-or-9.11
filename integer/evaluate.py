from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
from openai import OpenAI
from tqdm import tqdm 
import concurrent.futures
import os
def evaluate_binary_classification(y_true, y_pred):
    """
    评估二分类任务的预测结果
    :param y_true: 真实标签（0或1）
    :param y_pred: 模型预测标签（0或1）
    :return: 各种评价指标
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label='A')
    recall = recall_score(y_true, y_pred, pos_label='A')
    f1 = f1_score(y_true, y_pred, pos_label='A')
    cm = confusion_matrix(y_true, y_pred)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist()
    }

    return results

client = OpenAI(api_key="sk-599eb172897344d0ba73ac3548c1e50a", base_url="https://api.deepseek.com")
def chat_with_model(input_text, temperature=1.25):
    # 调用OpenAI API的chat接口
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个友好的助手。"},
            {"role": "user", "content": input_text}
        ],
        stream=False,
        temperature=temperature,
    )
    return response.choices[0].message.content


def process_single_entry(index, entry):
    question = f"谁更大？A: {entry['A']} B: {entry['B']}"
    answer = chat_with_model(question)
    predicted_label = 'A' if 'A' in answer else 'B'
    result_entry = {
        '关键词': entry.get('关键词', '整数比较'),
        '整数位数': entry.get('整数位数', None),
        '小数位数': entry.get('小数位数', None),
        'A': entry['A'],
        'B': entry['B'],
        '标签': entry['标签'],
        '预测': predicted_label
    }
    return index, result_entry, entry['标签'], predicted_label

# 加载数据集并进行测试
def load_and_test_dataset(file_path, output_file):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    y_true = []
    y_pred = []
    results = [None] * len(data)  # 预先分配列表以保持顺序

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_single_entry, idx, entry): idx for idx, entry in enumerate(data)}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(data), desc=f"Processing {file_path}"):
            idx, result_entry, true_label, predicted_label = future.result()
            results[idx] = (result_entry, true_label, predicted_label)

    # 保存中间结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for result_entry, true_label, predicted_label in results:
            y_true.append(true_label)
            y_pred.append(predicted_label)
            f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

    return y_true, y_pred

if __name__ == "__main__":
    input_dir = 'integer\data'
    output_dir = 'integer\output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_paths = [
        'two_digit_comparisons.json'
    ]
    
    for file_path in file_paths:
        input_file = os.path.join(input_dir, file_path)
        output_file = os.path.join(output_dir, file_path.replace('.json', '_results_t=1.25.json'))
        y_true, y_pred = load_and_test_dataset(input_file, output_file)
        results = evaluate_binary_classification(y_true, y_pred)
        
        # 保存最终评价结果
        final_output_file = os.path.join(output_dir, file_path.replace('.json', '_final_results_t=1.25.json'))
        with open(final_output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"测试结果已保存到 {final_output_file}")

