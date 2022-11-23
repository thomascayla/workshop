# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
%matplotlib inline
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import matplotlib.pyplot as plt
import os
import seaborn as sns

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
transactions_known_scored = dataiku.Dataset("transactions_known_scored").get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
list_var = ['authorized_flag','proba_1']
transactions_known_scored = transactions_known_scored[list_var]
transactions_known_scored['decile'] = pd.cut(x=transactions_known_scored.proba_1,
                                             bins=10, precision=1, right=False)
transactions_known_scored.decile = transactions_known_scored.decile.astype(str)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
col_names = {'authorized_flag':'sum_true', 'proba_1':'count_pred'}
df_by_deciles = transactions_known_scored.groupby(by='decile').agg({'authorized_flag':'sum',
                                                                    'proba_1':'count'}).rename(columns=col_names)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_by_deciles['validation_ratio'] = df_by_deciles.sum_true/df_by_deciles.count_pred
df_by_deciles.sort_index(ascending=True, inplace=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
sns.set_style("ticks",{'axes.grid' : True})
colors = ['#377eb8', '#4daf4a']
labels = list(df_by_deciles.index)
sum_true = list(df_by_deciles.sum_true)
count_pred = list(df_by_deciles.count_pred)
validation_ratio = list(df_by_deciles.validation_ratio)
width = 2/3 # the width of the bars

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.bar(labels, count_pred, width, label='count_pred', color=colors[0])
ax1.bar(labels, sum_true, width, bottom=count_pred, label='sum_true', color=colors[1])
ax1.set_ylabel('Number of predictions vs true labels', color=colors[0])
ax1.tick_params('y', colors=colors[0])
ax1.legend(loc='upper left')
ax1.set_xlabel('Probability estimates')

# .patches is everything inside of the chart
for rect in ax1.patches:
    # Find where everything is located
    height = rect.get_height()
    width = rect.get_width()
    x = rect.get_x()
    y = rect.get_y()

    # The height of the bar is the data value and can be used as the label
    label_text = f'{height:,.0f}' #to format decimal values

    # ax1.text(x, y, text)
    label_x = x + width / 2
    label_y = y + height / 2
    ax1.text(label_x, label_y, label_text, ha='center', va='center', fontsize=12, color='w', weight='bold')


ax2 = ax1.twinx()
color = 'r'
ax2.plot(labels, validation_ratio, label='True ratio', color=color)
ax2.set_ylabel('Percentage = sum_true / count_pred', color=color)
ax2.tick_params('y', colors=color)
ax2.legend(loc='upper right')

# Add ratio value to each bar.
for i, ratio in enumerate(validation_ratio):
    ax2.text(df_by_deciles.validation_ratio.index[i], ratio, '{:,.0%}'.format(ratio), ha='center',
             weight='bold', fontsize=12)


fig.tight_layout()
plt.title('True vs predicted observations by decile of probas (on the validation set)', fontsize=16)
plt.show()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
## Vérifier qu'on a les bons volumes (somme notamment) et commencer une review à présenter
## Dans quelle table le mettre
## Subject matter: all kind of binary classification (such as churn or engagement scoring)
## Join legend

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Save plot to folder
folder_for_plot = dataiku.Folder("ofSVU3Pe")
folder_path = folder_for_plot.get_path()

path_fig = os.path.join(folder_path, "output.png")
plt.savefig(path_fig)