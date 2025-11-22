'''
Description:
    This code is trying to plot the correlation heatmap between 32-channel eeg signal and different traffix contexts.
    - each row is diff traffic conditions
        1：Town06-rainy-nojam
        2：Town06-rainy-nojam
        3：Town06-sunny-jam
        4：Town06-sunny-nojam
        5：Town10-rainy-jam
        6：Town10-rainy-nojam
        7：Town10-sunny-jam
        8：Town10-sunny-nojam
        (Total: 8)

    - each column is diff eeg electro signal
        'Cz'	'FZ'	'FP1'	'F7'	'F3'	'FC1'	'C3'	'FC5'	'FT9'	'T7'	'CP5'	'CP1'	'P3'	'P7'
        'PO9'	'O1'	'PZ'	'Oz'	'O2'	'PO10'	'P8'	'P4'	'CP2'	'CP6'	'T8'	'FT10'	'FC6'	'C4'
        'FC2'	'F4'	'F8'	'FP2'
Author:
    CHAI Bo & FANG Le (Leo)
'''

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr

# set globe style
plt.rcParams.update({
    'axes.titlesize': 10,         # Title font size
    'axes.labelsize': .5,         # Axis label font size
    'xtick.labelsize': 6,         # X-axis tick label font size
    'ytick.labelsize': 6,         # Y-axis tick label font size
    'lines.linewidth': .5,        # Line width
    'lines.color': 'b',           # Line color
    'figure.figsize': (15, 3),    # Figure size
    'axes.grid': False,           # Enable grid
    'grid.alpha': 0.3,            # Grid transparency
})

# list task names and electro names
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

elec_name = [
    'Cz', 'FZ', 'FP1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5',
    'CP1', 'P3', 'P7', 'PO9', 'O1', 'PZ', 'Oz', 'O2', 'PO10', 'P8', 'P4',
    'CP2', 'CP6', 'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'FP2'
]

# generate dataframe to store final mean and variance values
Final_Mean = pd.DataFrame({('Task',''):range(1,9)})
Final_Var = pd.DataFrame({('Task',''):range(1,9)})

# breakpoint()


# participant NO
for part_NO in range(1,51):
    folder_path = f"path\\to\\Clip_EEG_noArtifact_icaCUS_thr100\\P{part_NO}\\"

    for filename in os.listdir(folder_path):

        match = re.match(r'(\d{4})_p\d+_(\d+)', filename)
        data = match.group(1)
        task = match.group(2)
        # breakpoint()

        try:
            df = pd.read_csv(folder_path+filename)
            EEG = df.iloc[:, 2:]

            for col in EEG.columns:
                filtered_values = EEG[col]
                Final_Mean.loc[int(task)-1, (f'P{part_NO}',col)] = filtered_values.mean()
                Final_Var.loc[int(task)-1, (f'P{part_NO}', col)] = filtered_values.var()

        except pd.errors.EmptyDataError:
            print(f'{filename}  is empty!')
            continue

# compute Spearman Correlation
Final_Mean = Final_Mean.dropna(axis=1)
Final_Var = Final_Var.dropna(axis=1)
traffic_contexts = {
    'Area': ['suburban', 'urban'],
    'Weather': ['rainy', 'sunny'],
    'Traffic': ['flow', 'jam']
}

corr_map_var = pd.DataFrame(None)

for elec in EEG.columns:  # elec_name
    # eeg chanel signal variance
    EEG_var = Final_Var.xs(elec, level=1, axis=1)
    EEG_var = EEG_var.reset_index().melt(id_vars='index', var_name='Column', value_name='Value')
    data_var = EEG_var['Value'].tolist()
    label_ind_var = EEG_var['index'].tolist()

    for category, value in traffic_contexts.items():
        label_var = [0 if value[0] in task_name[l] else 1 for ind, l in enumerate(label_ind_var)]
        corr, p_value = pointbiserialr(data_var, label_var)
        corr_map_var.loc[category, elec] = corr


plt.figure()
sns.heatmap(corr_map_var, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('EEG Variance')

plt.show()

breakpoint()














