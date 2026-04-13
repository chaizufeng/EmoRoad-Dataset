'''
Description:
    This code is trying to plot the Correlation heatmap between EmoSense and different traffic contexts
    - each row is diff traffic conditions
        1：Area: suburban/urban
        2: Weather: rainy/sunny
        3: Traffic: flow/jam

    - each column is EmoSense variables
        Frequency (Index) Corresponding to the Maximum of EmoSense Spectrum
        Magnitude Corresponding to the Maximum of EmoSense Spectrum

Author:
    CHAI Bo & FANG Le (Leo)
'''

import os
import re
import pandas as pd
import csv
import chardet
import matplotlib.pyplot as plt
import ast
from scipy.stats import spearmanr, pointbiserialr
import seaborn as sns
import numpy as np

# Set global style
plt.rcParams.update({
    'axes.titlesize': 10,    # Title font size
    'axes.labelsize': .5,     # Axis label font size
    'xtick.labelsize': 6,    # X-axis tick label font size
    'ytick.labelsize': 6,    # Y-axis tick label font size
    'lines.linewidth': .5,    # Line width
    'lines.color': 'b',      # Line color
    'figure.figsize': (15, 3), # Figure size
    'axes.grid': False,       # Enable grid
    'grid.alpha': 0.3,       # Grid transparency
})


# list task names & EmoSense variables
task_name = [
    'suburban-sunny-flow',
    'suburban-rainy-flow',
    'suburban-sunny-jam',
    'suburban-rainy-jam',
    'urban-sunny-flow',
    'urban-rainy-flow',
    'urban-sunny-jam',
    'urban-rainy-jam',
]

EmoSense_features = [
    'Frequency',
    'Magnitude'
]

# generate dataframe to store final mean and variance values
Final_Mean = pd.DataFrame({('Task',''):range(1,9)})
Final_Var = pd.DataFrame({('Task',''):range(1,9)})


# participant NO
for part_NO in range(1,51):
    folder_path = f'path\\to\\Clip_EmoSense\\P{part_NO}\\'

    for filename in os.listdir(folder_path):

        match = re.match(r'(\d{4})_p\d+_(\d+)', filename)
        date = match.group(1)
        task = match.group(2)

        try:
            df = pd.read_csv(folder_path+filename,skiprows=1,usecols=range(1,150),header=None)
            emo_spec = df.apply(pd.to_numeric,errors='coerce').dropna(axis=1)
            # emo_spec = 20 * np.log10(emo_spec)
            max_col = emo_spec.idxmax(axis=1)
            max_val = emo_spec.max(axis=1)
            max_col = np.diff(max_col)
            max_val = np.diff(max_val)
            Final_Mean.loc[int(task)-1, (f'P{part_NO}','Frequency')] = max_col.mean()
            Final_Mean.loc[int(task) - 1, (f'P{part_NO}', 'Magnitude')] = max_val.mean()
            Final_Var.loc[int(task) - 1, (f'P{part_NO}', 'Frequency')] = max_col.var()
            Final_Var.loc[int(task) - 1, (f'P{part_NO}', 'Magnitude')] = max_val.var()

        except pd.errors.EmptyDataError:
            print(f'{filename} is empty!')
            continue


# breakpoint()

# compute Correlation
Final_Mean = Final_Mean.dropna(axis=1)
Final_Var = Final_Var.dropna(axis=1)
traffic_contexts = {
    'Area': ['suburban', 'urban'],
    'Weather': ['rainy', 'sunny'],
    'Traffic': ['flow', 'jam']
}

corr_map_mean = pd.DataFrame(None)
corr_map_var = pd.DataFrame(None)

for EmoSense_feature in EmoSense_features:
    EmoSense_mean = Final_Mean.xs(EmoSense_feature,level=1,axis=1)
    EmoSense_mean = EmoSense_mean.reset_index().melt(id_vars='index', var_name='Column', value_name='Value')
    data_mean = EmoSense_mean['Value'].tolist()
    # data_mean = 10*np.log10(data_mean)
    label_ind_mean = EmoSense_mean['index'].tolist()

    EmoSense_var = Final_Var.xs(EmoSense_feature, level=1, axis=1)
    EmoSense_var = EmoSense_var.reset_index().melt(id_vars='index', var_name='Column', value_name='Value')
    data_var = EmoSense_var['Value'].tolist()
    # data_var = 10*np.log10(data_var)
    label_ind_var = EmoSense_var['index'].tolist()

    for category, value in traffic_contexts.items():
        label_mean = [0 if value[0] in task_name[l] else 1 for ind, l in enumerate(label_ind_mean)]
        # corr, p_value = spearmanr(data_mean, label_mean)
        corr, p_value = pointbiserialr(data_mean, label_mean)
        corr_map_mean.loc[category,EmoSense_feature + ' Corresponding to the Maximum of EmoSense Spectrum'] = corr

        label_var = [0 if value[0] in task_name[l] else 1 for ind, l in enumerate(label_ind_var)]
        # corr, p_value = spearmanr(data_var, label_var)
        corr, p_value = pointbiserialr(data_var, label_var)
        corr_map_var.loc[category, EmoSense_feature + ' Corresponding to the Maximum of EmoSense Spectrum'] = corr

plt.figure(1)
sns.heatmap(corr_map_mean, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('EmoSense Mean')

plt.figure(2)
sns.heatmap(corr_map_var, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('EmoSense Variance')

plt.show()


breakpoint()






