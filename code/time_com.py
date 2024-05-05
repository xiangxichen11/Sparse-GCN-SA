import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, LogFormatter

# Data
models = ['GCN', 'GCN-SA', 'GCN-SA + BBird', 'GCN-SA + Exph']
datasets = ['Texas', 'Citerseer', 'Cora', 'Pubmed', 'Charmeleon', 'Squirrel']
running_time = {
    'GCN': [2.3, 2.5, 2, 11, 1.8, 3],
    'GCN-SA': [10.2, 104.9, 68.3, 200.7, 32.5, 289.4],
    'GCN-SA + BBird': [10.4, 98.6, 72.1, 197.2, 34.3, 283.6], 
    'GCN-SA + Exph': [9.2, 123.4, 68.3, 182.7, 29.9, 277.4]
}

# Plotting
fig, ax = plt.subplots(figsize=(12, 3.8))  # Wider figure

bar_width = 0.2
index = range(len(datasets))

colors = ['#87cfeb', '#D587EB', '#EBA387', '#9DDB8C']  # Custom colors

for i, model in enumerate(models):
    plt.bar([p + i * bar_width for p in index], running_time[model], bar_width, label=model, color=colors[i], edgecolor='black', linewidth=0.5)

plt.ylabel('Running time of 500 epochs (seconds)', fontsize=12)

# Set y-axis scale to log
plt.yscale('log')

# Customize y-axis ticks
plt.gca().yaxis.set_major_locator(LogLocator(subs=[1.0, 10.0]))  # Set base to 10
formatter = LogFormatter(labelOnlyBase=False)
formatter.create_dummy_axis()
plt.gca().yaxis.set_major_formatter(formatter)

# Manually adjust the position of the y=10 tick
ax.yaxis.get_major_ticks()[1].set_pad(15)

# Set y-axis limit to 1000 and start from 1
plt.ylim(1, 1000)

# Only show axis on left and bottom
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xticks([p + bar_width * (len(models) - 1) / 2 for p in index], datasets, rotation=0, ha='center', fontsize=10)  # Horizontal x labels parallel to x-axis
legend = plt.legend(edgecolor='black', fancybox=False)  # Add border to legend colors and set font

# Save figure as PDF
plt.savefig('time_complexity.pdf', bbox_inches='tight')

plt.show()
