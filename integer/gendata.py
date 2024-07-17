import json
import os

# 生成所有两位数的比较数据集
def generate_two_digit_comparisons():
    data = []
    for integer_a in range(10, 100):  # 遍历两位数的整数部分
        for integer_b in range(10, 100):
            if integer_a > integer_b:
                label = 'A'
                question = {
                    "关键词": "两位整数比较",
                    "整数位数": 2,
                    "A": integer_a,
                    "B": integer_b,
                    "标签": label
                }
                data.append(question)
    return data

# 保存为 JSON 文件
def save_comparisons_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"所有两位整数比较数据已保存到 {filename} 文件")

# 生成并保存数据
if __name__ == "__main__":
    output_dir = 'integer/data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, 'two_digit_comparisons.json')
    comparisons = generate_two_digit_comparisons()
    save_comparisons_to_json(comparisons, output_file)
