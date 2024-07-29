import json

# 生成类似 9.9 和 9.11 的所有组合
def generate_float_comparisons():
    data = []
    for i in range(1, 11):
        i += 1
        for integer_part in range(0, 10):
            for decimal_part_a in range(2, 10):
                a = f"{integer_part}.{decimal_part_a}"
                b = f"{integer_part}.11"
                if float(a) > float(b):
                    label = 'A'
                    question = {
                        "value of the integer part": integer_part,
                        "A": a,
                        "B": b,
                        "Label": label
                    }
                    data.append(question)
    return data

# 保存为 JSON 文件
def save_comparisons_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"all data save in {filename} !!")

# 生成并保存数据
if __name__ == "__main__":
    comparisons = generate_float_comparisons()
        
    save_comparisons_to_json(comparisons, r'data\greater_than_11_comparisons_10.json')
