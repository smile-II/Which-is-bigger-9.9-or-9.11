import random
import json

# 生成整数比较数据
def generate_integer_data(num_samples=20, digits=1):
    data = []
    for _ in range(num_samples):
        a = random.randint(10**(digits-1), 10**digits - 1)
        b = random.randint(10**(digits-1), 10**digits - 1)
        label = 'A' if a > b else ('B' if b > a else 'C')
        question = {
            "整数位数": digits,
            "A": a,
            "B": b,
            "标签": label
        }
        data.append(question)
    return data

# 生成所有位数的整数比较数据并保存为JSON文件
def generate_and_save_all_integer_data():
    all_data = []
    for digits in range(1, 6):
        data = generate_integer_data(20, digits)
        all_data.extend(data)

    with open('integer_comparison_data.json', 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
    print("所有整数比较数据已保存到 integer_comparison_data.json 文件")

# 运行代码生成并保存数据
# generate_and_save_all_integer_data()


# 生成固定小数位数的浮点数
def generate_fixed_decimal_float(digits, decimal_places):
    format_string = f"{{:.{decimal_places}f}}"
    return format_string.format(round(random.uniform(0.1, 1), decimal_places))

# 生成小数比较数据
def generate_float_data(num_samples=20, decimal_places=1):
    data = []
    for _ in range(num_samples):
        a = generate_fixed_decimal_float(decimal_places, decimal_places)
        b = generate_fixed_decimal_float(decimal_places, decimal_places)
        label = 'A' if float(a) > float(b) else ('B' if float(b) > float(a) else 'C')
        question = {
            "关键词": "小数比较",
            "小数位数": decimal_places,
            "A": a,
            "B": b,
            "标签": label
        }
        data.append(question)
    return data

# 生成所有位数的小数比较数据并保存为JSON文件
def generate_and_save_all_float_data():
    all_data = []
    for decimal_places in range(1, 6):
        data = generate_float_data(20, decimal_places)
        all_data.extend(data)

    with open('float_comparison_data.json', 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
    print("所有小数比较数据已保存到 float_comparison_data.json 文件")

# 运行代码生成并保存数据
# generate_and_save_all_float_data()



# 生成固定整数和小数位数的小数
def generate_fixed_decimal_float(integer_value, decimal_places):
    decimal_part = random.randint(10**(decimal_places-1), 10**decimal_places - 1)
    format_string = f"{{0}}.{{1:0{decimal_places}d}}"
    return format_string.format(integer_value, decimal_part)

# 生成小数比较数据
def generate_large_float_data(num_samples=20, integer_value=1, decimal_places=1):
    data = []
    for _ in range(num_samples):
        a = generate_fixed_decimal_float(integer_value, decimal_places)
        b = generate_fixed_decimal_float(integer_value, decimal_places)
        while a == b:  # 确保A和B不同
            b = generate_fixed_decimal_float(integer_value, decimal_places)
        label = 'A' if float(a) > float(b) else 'B'
        question = {
            "关键词": "大于1的小数比较",
            "整数位数": len(str(integer_value)),
            "小数位数": decimal_places,
            "A": a,
            "B": b,
            "标签": label
        }
        data.append(question)
    return data

# 生成所有位数的小数比较数据并保存为JSON文件
def generate_and_save_all_large_float_data():
    all_data = []
    for integer_digits in range(1, 4):  # 整数位数从1到3
        integer_value = random.randint(10**(integer_digits-1), 10**integer_digits - 1)
        for decimal_places in range(1, 5):  # 小数位数从1到4
            data = generate_large_float_data(20, integer_value, decimal_places)
            all_data.extend(data)

    with open('large_float_comparison_data.json', 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
    print("所有大于1的小数比较数据已保存到 large_float_comparison_data.json 文件")

# 运行代码生成并保存数据
# generate_and_save_all_large_float_data()






# 生成特定类型的小数对
def generate_special_float_pair(integer_value, decimal_places_few, decimal_places_many):
    # 小数位数少，但十分位大
    decimal_part_few = random.randint(5, 9) * (10 ** (decimal_places_few - 1))
    a = f"{integer_value}.{decimal_part_few:0{decimal_places_few}d}"
    
    # 小数位数多，但十分位不能为0且小于少位数的数的十分位
    decimal_part_many_first_digit = random.randint(1, 4)
    decimal_part_many_remainder = random.randint(0, 10**(decimal_places_many - 1) - 1)
    decimal_part_many = decimal_part_many_first_digit * (10 ** (decimal_places_many - 1)) + decimal_part_many_remainder
    b = f"{integer_value}.{decimal_part_many:0{decimal_places_many}d}"

    return a, b

# 生成小数比较数据
def generate_special_float_data(num_samples, integer_value, decimal_places_few, decimal_places_many):
    data = []
    for _ in range(num_samples):
        a, b = generate_special_float_pair(integer_value, decimal_places_few, decimal_places_many)
        label = 'A' if float(a) > float(b) else 'B'
        question = {
            "关键词": "特定小数比较",
            "整数位数": len(str(integer_value)),
            "小数位数少": decimal_places_few,
            "小数位数多": decimal_places_many,
            "A": a,
            "B": b,
            "标签": label
        }
        data.append(question)
    return data

# 生成小数补零对齐数据
def pad_zeros(data):
    padded_data = []
    for entry in data:
        a = entry["A"]
        b = entry["B"]
        decimal_places_many = entry["小数位数多"]
        padded_a = f"{float(a):.{decimal_places_many}f}"
        padded_entry = {
            "关键词": entry["关键词"],
            "整数位数": entry["整数位数"],
            "小数位数少": entry["小数位数少"],
            "小数位数多": decimal_places_many,
            "A": padded_a,
            "B": b,
            "标签": entry["标签"]
        }
        padded_data.append(padded_entry)
    return padded_data

# 生成所有特定类型的小数比较数据并保存为JSON文件
def generate_and_save_all_special_float_data():
    all_data = []
    for integer_digits in range(1, 3):  # 整数位数从1到2
        for decimal_places_few in range(1, 5):  # 小数位数少从1到4
            for decimal_places_many in range(decimal_places_few + 1, 6):  # 小数位数多从2到5
                integer_value = random.randint(10**(integer_digits-1), 10**integer_digits - 1)
                data = generate_special_float_data(20, integer_value, decimal_places_few, decimal_places_many)
                all_data.extend(data)

    # 保存原始数据
    with open('special_float_comparison_data.json', 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
    print("所有特定小数比较数据已保存到 special_float_comparison_data.json 文件")

    # 生成补零对齐数据并保存
    padded_data = pad_zeros(all_data)
    with open('special_float_comparison_data_padded.json', 'w', encoding='utf-8') as f:
        json.dump(padded_data, f, ensure_ascii=False, indent=4)
    print("所有特定小数补零对齐数据已保存到 special_float_comparison_data_padded.json 文件")

# 运行代码生成并保存数据
# generate_and_save_all_special_float_data()





# 生成固定整数和小数位数的小数
def generate_fixed_decimal_float(integer_value, decimal_places):
    decimal_part = random.randint(10**(decimal_places-1), 10**decimal_places - 1)
    format_string = f"{{0}}.{{1:0{decimal_places}d}}"
    return format_string.format(integer_value, decimal_part)

# 生成小数比较数据
def generate_large_float_data(num_samples=20, integer_value=1, decimal_places=1):
    data = []
    for _ in range(num_samples):
        a = generate_fixed_decimal_float(integer_value, decimal_places)
        b = generate_fixed_decimal_float(integer_value, decimal_places)
        while a == b:  # 确保A和B不同
            b = generate_fixed_decimal_float(integer_value, decimal_places)
        label = 'A' if float(a) > float(b) else 'B'
        question = {
            "关键词": "大于1的小数比较",
            "整数位数": len(str(integer_value)),
            "小数位数": decimal_places,
            "A": a,
            "B": b,
            "标签": label
        }
        data.append(question)
    return data

# 生成小数补零对齐数据
def pad_zeros(data):
    padded_data = []
    for entry in data:
        a = entry["A"]
        b = entry["B"]
        decimal_places = entry["小数位数"]
        
        # 比较a和b的大小并给较小的数补零
        if float(a) < float(b):
            padded_a = f"{a}0"
            padded_b = b
        else:
            padded_a = a
            padded_b = f"{b}0"
        
        padded_entry = {
            "关键词": entry["关键词"],
            "整数位数": entry["整数位数"],
            "小数位数": entry["小数位数"],
            "A": padded_a,
            "B": padded_b,
            "标签": entry["标签"]
        }
        padded_data.append(padded_entry)
    return padded_data
    

# 生成所有位数的小数比较数据并保存为JSON文件
def generate_and_save_all_large_float_data():
    all_data = []
    for integer_digits in range(1, 4):  # 整数位数从1到3
        for decimal_places in range(1, 5):  # 小数位数从1到4
            for _ in range(20):  # 每种组合生成20个样本
                integer_value = random.randint(10**(integer_digits-1), 10**integer_digits - 1)
                data = generate_large_float_data(1, integer_value, decimal_places)
                all_data.extend(data)

    # 保存原始数据
    with open('large_float_comparison_data.json', 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
    print("所有大于1的小数比较数据已保存到 large_float_comparison_data.json 文件")

    # 生成补零对齐数据并保存
    padded_data = pad_zeros(all_data)
    with open('large_float_comparison_data_padded.json', 'w', encoding='utf-8') as f:
        json.dump(padded_data, f, ensure_ascii=False, indent=4)
    print("所有大于1的小数补零对齐数据已保存到 large_float_comparison_data_padded.json 文件")

# 运行代码生成并保存数据
# generate_and_save_all_large_float_data()


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
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"所有小数比较数据已保存到 {filename} 文件")

# 生成并保存数据
if __name__ == "__main__":
    comparisons = generate_float_comparisons()
    save_comparisons_to_json(comparisons, 'data/float_comparisons.json')




