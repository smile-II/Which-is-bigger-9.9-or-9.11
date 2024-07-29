import torch
import json
import os
from tqdm import tqdm 
import concurrent.futures
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_responses(prompts, model, tokenizer, device):
    # Tokenize the input texts with attention_mask
    model = model.eval()
    model_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model.generate(**model_inputs, max_length=512,use_cache=False)
    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return responses

def evaluate_binary_classification(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == 'A')
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == 'B')
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'B' and yp == 'A')
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'A' and yp == 'B')

    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def process_batch(entries, model, tokenizer, device, prompt_template):
    prompts = [prompt_template.format(entry['A'], entry['B']) for entry in entries]
    answers = generate_responses(prompts, model, tokenizer, device)
    
    results = []
    for entry, answer in zip(entries, answers):
        predicted_label = 'B'
        for char in reversed(answer.strip()):
            if char == 'A' or char == 'B':
                predicted_label = char
                break
        result_entry = {
            'value of the integer part': entry.get('value of the integer part', None),
            'A': entry['A'],
            'B': entry['B'],
            'Label': entry['Label'],
            'Prediction': predicted_label,
            'Answer': answer
        }
        results.append((result_entry, entry['Label'], predicted_label))
    
    return results

def load_and_test_dataset(file_path, output_file, model, tokenizer, device, prompt_template, batch_size=16):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    y_true = []
    y_pred = []

    results = []
    for i in tqdm(range(0, len(data), batch_size), desc=f"Processing {file_path}"):
        batch_entries = data[i:i + batch_size]
        batch_results = process_batch(batch_entries, model, tokenizer, device, prompt_template)
        results.extend(batch_results)
    
    for result_entry, true_label, predicted_label in results:
        y_true.append(true_label)
        y_pred.append(predicted_label)
        
        # Append the result to the output file
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

    return y_true, y_pred

def main():
    input_dir = "/home/home/txs/sjl/gan/data/"  # 请替换为实际的输入目录路径
    output_dir = "/home/home/txs/sjl/gan/data/output/prompt_1"  # 请替换为实际的输出目录路径
    model_path = "/home/home/txs/sjl/model/glm3"  # 请替换为实际的模型路径
    device = "cuda:0"  # 或者 "cpu" 根据你的环境选择
    prompt_idx = 1  # 根据需要选择提示词的索引
    file_paths = ["filtered_errors.json"]  # 请替换为实际的文件名列表

    prompts = [
        "Which of these two numbers is larger? A: {} B: {}. \n###You must answer only with the single letter 'A' or 'B'. \n###You can only output a letter. \n###You output letter is:",
        "Which is larger? A: {} B: {} ## Please output only option A or option B.", 
        "Which of these two numbers is larger? A: {} B: {} ## Please output only option A or option B.",
        "Which of these two numbers is larger? A: {} B: {} ## Please think step by step, and output your option A or B for the last letter.",
    ]

    prompt_template = prompts[prompt_idx]

    # Load the model and tokenizer once
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.half()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file_path in file_paths:
        input_file = os.path.join(input_dir, file_path)
        output_file = os.path.join(output_dir, file_path.replace('.json', '_results.json'))

        # Clear the output file before starting
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('')

        y_true, y_pred = load_and_test_dataset(input_file, output_file, model, tokenizer, device, prompt_template, batch_size=16)
        results = evaluate_binary_classification(y_true, y_pred)
        
        final_output_file = os.path.join(output_dir, file_path.replace('.json', '_final_results.json'))
        with open(final_output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"测试结果已保存到 {final_output_file}")

if __name__ == "__main__":
    main()
