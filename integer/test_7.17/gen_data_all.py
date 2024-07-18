import random
import pandas as pd
from datetime import datetime, timedelta

# 生成整数比较数据
def generate_integer_data(n=100):
    data = []
    for _ in range(n):
        a, b = random.randint(1, 1000), random.randint(1, 1000)
        data.append((a, b, '整数'))
    return data

# 生成浮点数比较数据
def generate_float_data(n=100):
    data = []
    for _ in range(n):
        a, b = round(random.uniform(1, 1000), 2), round(random.uniform(1, 1000), 2)
        data.append((a, b, '浮点数'))
    return data

# 生成二进制数比较数据
def generate_binary_data(n=100):
    data = []
    for _ in range(n):
        a, b = bin(random.randint(1, 1000))[2:], bin(random.randint(1, 1000))[2:]
        data.append((a, b, '二进制数'))
    return data

# 生成十六进制数比较数据
def generate_hex_data(n=100):
    data = []
    for _ in range(n):
        a, b = hex(random.randint(1, 1000))[2:], hex(random.randint(1, 1000))[2:]
        data.append((a, b, '十六进制数'))
    return data

# 生成日期和时间比较数据
def generate_date_data(n=100):
    data = []
    start_date = datetime.now()
    for _ in range(n):
        a = start_date + timedelta(days=random.randint(0, 365))
        b = start_date + timedelta(days=random.randint(0, 365))
        data.append((a.strftime('%Y-%m-%d'), b.strftime('%Y-%m-%d'), '日期'))
    return data

# 生成软件版本号比较数据
def generate_version_data(n=100):
    data = []
    for _ in range(n):
        a = f"{random.randint(0, 2)}.{random.randint(0, 9)}.{random.randint(0, 9)}"
        b = f"{random.randint(0, 2)}.{random.randint(0, 9)}.{random.randint(0, 9)}"
        data.append((a, b, '软件版本号'))
    return data

# 生成科学记数法比较数据
def generate_scientific_data(n=100):
    data = []
    for _ in range(n):
        a = f"{random.uniform(1, 1000):.2e}"
        b = f"{random.uniform(1, 1000):.2e}"
        data.append((a, b, '科学记数法'))
    return data

# 生成货币值比较数据
def generate_currency_data(n=100):
    data = []
    for _ in range(n):
        a, b = f"${random.uniform(1, 1000):.2f}", f"${random.uniform(1, 1000):.2f}"
        data.append((a, b, '货币值'))
    return data

# 生成混合表示比较数据
def generate_mixed_data(n=100):
    data = []
    for _ in range(n):
        a = random.choice([bin(random.randint(1, 1000))[2:], hex(random.randint(1, 1000))[2:], f"{random.uniform(1, 1000):.2e}"])
        b = random.choice([bin(random.randint(1, 1000))[2:], hex(random.randint(1, 1000))[2:], f"{random.uniform(1, 1000):.2e}"])
        data.append((a, b, '混合表示'))
    return data

# 合并所有数据集
def generate_all_data():
    data = []
    data.extend(generate_integer_data())
    data.extend(generate_float_data())
    data.extend(generate_binary_data())
    data.extend(generate_hex_data())
    data.extend(generate_date_data())
    data.extend(generate_version_data())
    data.extend(generate_scientific_data())
    data.extend(generate_currency_data())
    data.extend(generate_mixed_data())
    return data

# 生成数据集并保存为CSV文件
data = generate_all_data()
df = pd.DataFrame(data, columns=['值1', '值2', '类型'])
df.to_csv('comparison_dataset.csv', index=False)
print("数据集生成完成并保存为comparison_dataset.csv文件")

