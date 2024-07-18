import argparse
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from openai import OpenAI
from tqdm import tqdm 
import concurrent.futures

def evaluate_binary_classification(y_true, y_pred):
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

def chat_with_model(input_text, temperature=1):
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

def process_single_entry(index, entry, temperature, prompt):
    question = prompt.format(entry['A'], entry['B'])
    answer = chat_with_model(question, temperature)
    predicted_label = 'A' if 'A' in answer else 'B'
    result_entry = {
        'value of the integer part': entry.get('value of the integer part', None),
        # 'Number of fewer decimal places': entry.get('Number of fewer decimal places', None),
        # 'Number of more decimal places': entry.get('Number of more decimal places', None),
        'A': entry['A'],
        'B': entry['B'],
        'Label': entry['Label'],
        'Prediction': predicted_label,
        'Answer': answer
    }
    return index, result_entry, entry['Label'], predicted_label


def load_and_test_dataset(file_path, output_file, temperature, prompt):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    y_true = []
    y_pred = []
    results = [None] * len(data)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_single_entry, idx, entry, temperature, prompt): idx for idx, entry in enumerate(data)}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(data), desc=f"Processing {file_path} with temperature {temperature}"):
            idx, result_entry, true_label, predicted_label = future.result()
            results[idx] = (result_entry, true_label, predicted_label)

    with open(output_file, 'w', encoding='utf-8') as f:
        for result_entry, true_label, predicted_label in results:
            y_true.append(true_label)
            y_pred.append(predicted_label)
            f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

    return y_true, y_pred

def main(input_dir, output_dir, prompt_idx, repetitions, temperatures):
    prompts = [
        "Which is larger? A: {} B: {}   ## Please output only option A or option B.",
        "Which of these two numbers is larger?  A: {} B: {}   ## Please output only option A or option B.",
        "Which of these two numbers is larger?  A: {} B: {}   ## Please think step by step, and output your option A or B for the last letter."
    ]

    prompt = prompts[prompt_idx]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_paths = [
        '0-9.json'
    ]

    for i in range(repetitions):
        for file_path in file_paths:
            input_file = os.path.join(input_dir, file_path)
            for temp in temperatures:
                output_file = os.path.join(output_dir, file_path.replace('.json', f'_results_t={temp}-{i}.json'))
                y_true, y_pred = load_and_test_dataset(input_file, output_file, temp, prompt)
                results = evaluate_binary_classification(y_true, y_pred)
                
                final_output_file = os.path.join(output_dir, file_path.replace('.json', f'_final_results_t={temp}-{i}.json'))
                with open(final_output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                print(f"测试结果已保存到 {final_output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model with different prompts")
    parser.add_argument("--input_dir", type=str, default=r'data', help="Input directory for the dataset")
    parser.add_argument("--output_dir", type=str, default=r'output', help="Output directory for the results")
    parser.add_argument("--prompt_idx", type=int, default=0, choices=[0, 1, 2], help="Index of the prompt to use")
    parser.add_argument("--repetitions", type=int, default=1, help="Number of repetitions for the experiment")
    parser.add_argument("--temperatures", type=float, nargs='+', default=[1.25, 1, 0.75, 0.5, 0.25, 0], help="List of temperatures to use for the model")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = os.path.join(args.output_dir, f'prompt_{args.prompt_idx}')
    
    main(input_dir, output_dir, args.prompt_idx, args.repetitions, args.temperatures)
