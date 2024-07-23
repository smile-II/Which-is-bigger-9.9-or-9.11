import json
import pandas as pd
import matplotlib.pyplot as plt

# Read the filtered JSONL file
input_file = 'float/output/data_visualization/filtered_errors.json'

data = []
with open(input_file, 'r', encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line.strip()))

# Load data into a Pandas DataFrame
df = pd.DataFrame(data)

# Convert columns A and B to floats
df['A'] = df['A'].astype(float)
df['B'] = df['B'].astype(float)

# Create scatter plot by temperature
fig, ax = plt.subplots()
colors = {'0': 'r', '0.25': 'g', '0.5': 'b', '0.75': 'c', '1': 'm', '1.25': 'y'}

for temp in df['temperature'].unique():
    subset = df[df['temperature'] == temp]
    ax.scatter(subset['A'], subset['B'], label=f't={temp}', color=colors.get(temp, 'k'))

ax.legend()
ax.set_xlabel('A')
ax.set_ylabel('B')
ax.set_title('Scatter Plot of A and B Values for Incorrect Predictions (by Temperature)')

# Save the plot
plt.savefig('float/output/data_visualization/scatter_plot_by_temperature.png')

# Create scatter plot by prompt category
fig, ax = plt.subplots()
colors = {'prompt_0': 'r', 'prompt_1': 'g', 'prompt_2': 'b', 'prompt_3': 'c', 'prompt_4': 'm'}

for prompt in df['prompt'].unique():
    subset = df[df['prompt'] == prompt]
    ax.scatter(subset['A'], subset['B'], label=f'{prompt}', color=colors.get(prompt, 'k'))

ax.legend()
ax.set_xlabel('A')
ax.set_ylabel('B')
ax.set_title('Scatter Plot of A and B Values for Incorrect Predictions (by Prompt)')

# Save the plot
plt.savefig('float/output/data_visualization/scatter_plot_by_prompt.png')
