import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import  seaborn as sns

# dataset = '20ng'
# title = '{} dataset'.format(dataset)
# data = pd.read_csv('data', sep='\t', header=0)
# data = data.fillna('-')
# data = data[data['Name']==dataset]
# data['hue'] = data['methods'] + ':' + data['type']
# sns.set_style('whitegrid')
# sns.despine(trim=True)
# plt.title(title)
# sns.barplot(x='num_layer', y='best_test_acc', hue='hue', data=data)
#
# print(data)

sns.color_palette("Paired")
hue = ['red', 'blue','green']
sns.set_context("paper", font_scale=1.5)#, rc={"font.size":20,"axes.titlesize":15,"axes.labelsize":15})
# sns.set_style('whitegrid')

# fig = plt.Figure(figsize=(2,2))
# data = pd.read_csv('tr_split', sep='\t', header=None)
# data.columns = ['Training Percentage', 'Methods', 'Test Accuracy', 'Dataset']
# sns.lineplot(x=data.columns[0], y=data.columns[2], data=data, hue=data.columns[1],
#              markers=True, style=data.columns[1], palette=hue, linewidth=3, markersize=8)
# plt.grid()
# # plt.rc('xtick', labelsize=20)
# # plt.rc('ytick', labelsize=20)
#
# plt.savefig('tr_split_R52.pdf', dpi=500)
# plt.show()

data = pd.read_csv('tr_split', sep='\t', header=None)
data.columns = ['Macro-F1 score','Training Percentage', 'Methods', 'Micro-F1 score', 'Dataset', 'seed']
R52_data = data[data['Dataset']=='R52']
# R52_data['Methods'] = R52_data['Methods'].apply(lambda x: 'Ours' if x =='ours' else x)

# Ohsumed_data = data[data['Dataset']=='ohsumed']
fig, axes = plt.subplots(1, 2, figsize=(10,4))


a = sns.lineplot(ax=axes[0], x=data.columns[1], y=data.columns[3], data=R52_data, hue=data.columns[2],
             markers=True, style=data.columns[2], palette=hue, linewidth=2, markersize=7, legend=False)
# axes[0].legend(labels=['proposed methods', 'HyperGAT', 'TextING'], fontsize=15)
# axes[0].set_title('R52', fontsize=17, y=-0.23,ha='center')
axes[0].grid(True)

b = sns.lineplot(ax=axes[1], x=data.columns[1], y=data.columns[0], data=R52_data, hue=data.columns[2],
             markers=True, style=data.columns[2], palette=hue, linewidth=2, markersize=7)
# axes[1].legend(labels=['proposed methods', 'HyperGAT', 'TextING'], fontsize=15)
# axes[1].set_title('R52', fontsize=17, y=-0.3,ha='center')
#
axes[1].grid(True)
# handles, labels = axes[-1].get_legend_handles_labels()
# fig.legend(handles, labels, loc=2, ncol=4, borderaxespad=0, bbox_to_anchor=(1.05,1))
axes[1].legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.4), borderaxespad=0.5)
fig.tight_layout()
plt.savefig('tr_split.pdf', dpi=500)
plt.show()
