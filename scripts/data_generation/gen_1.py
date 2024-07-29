import json

# 生成类似 9.9 和 9.11 的所有组合


def generate_float_comparisons():
    data = []
    for integer_part in range(0, 10):
        # for decimal_part_a in range(1, 10):
        for decimal_part_a in range(10, 100, 10): #padded
            for decimal_part_b in range(1, 100):
                # 确保较小的数的小数部分的十分位不为0
                if decimal_part_b // 10 != 0:
                    a = f"{integer_part}.{decimal_part_a}"
                    b = f"{integer_part}.{decimal_part_b:02d}"
                    if float(a) > float(b):
                        label = 'A'
                        question = {
                            "value of the integer part": integer_part,
                            # "Number of fewer decimal places": len(str(decimal_part_a)),
                            # "Number of more decimal places": len(str(decimal_part_b)),
                            "A": a,
                            "B": b,
                            "Label": label

                        }
                        data.append(question)
    return data

# def generate_float_comparisons():
#     data = []
#     for integer_part in range(0, 10):
#         for decimal_part_a in range(0, 10):
#             for decimal_part_b in range(0, 100):
#                 a = f"{integer_part}.{decimal_part_a}"
#                 b = f"{integer_part}.{decimal_part_b:02d}"
#                 if float(a) != float(b):
#                     if float(a) > float(b):
#                         label = 'A'
#                     else:
#                         label = 'B'
#                     question = {
#                         "value of the integer part": integer_part,
#                         # "Number of fewer decimal places": len(str(decimal_part_a)),
#                         # "Number of more decimal places": len(str(decimal_part_b)),
#                         "A": a,
#                         "B": b,
#                         "Label": label
#                     }
#                     data.append(question)
#     return data

# 保存为 JSON 文件
def save_comparisons_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        # for entry in data:
        #     f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"all data save in {filename} !!")

# 生成并保存数据
if __name__ == "__main__":
    comparisons = generate_float_comparisons()
    save_comparisons_to_json(comparisons, 'float\data\\0-9_padded.json')
