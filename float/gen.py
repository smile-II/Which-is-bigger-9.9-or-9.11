import json

# 生成类似 9.9 和 9.11 的所有组合
def generate_float_comparisons():
    data = []
    for integer_part in range(1, 10):
        for decimal_part_a in range(1, 10):
            for decimal_part_b in range(1, 100):
                # 确保较小的数的小数部分的十分位不为0
                if decimal_part_b // 10 != 0:
                    a = f"{integer_part}.{decimal_part_a}"
                    b = f"{integer_part}.{decimal_part_b:02d}"
                    if float(a) > float(b):
                        label = 'A'
                        question = {
                            "关键词": "小数比较",
                            "整数位数": len(str(integer_part)),
                            "小数位数少": len(str(decimal_part_a)),
                            "小数位数多": len(str(decimal_part_b)),
                            "A": a,
                            "B": b,
                            "标签": label
                        }
                        data.append(question)
    return data

# 保存为 JSON 文件
def save_comparisons_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        # for entry in data:
        #     f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"所有小数比较数据已保存到 {filename} 文件")

# 生成并保存数据
if __name__ == "__main__":
    comparisons = generate_float_comparisons()
    save_comparisons_to_json(comparisons, 'float\data\\1.json')




