import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# 读取数据
df = pd.read_json('../../float/output/data_visualization/filtered_errors.json')

# 特征和标签
X = df[['A', 'B']]  # 假设特征为A和B
y = df['Label']

# 训练模型
model = RandomForestClassifier()
model.fit(X, y)

# 创建SHAP解释器
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# 绘制SHAP值
shap.summary_plot(shap_values, X)

# 分析高错误率B选项的SHAP值
high_error_B = df[df['B'] == '5.11']  # 假设5.11是高错误率B选项
shap_values_high_error = explainer(high_error_B[['A', 'B']])
shap.decision_plot(explainer.expected_value[1], shap_values_high_error[1], high_error_B[['A', 'B']])
