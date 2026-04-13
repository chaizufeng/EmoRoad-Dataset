'''
Description:
    This code is trying to plot the Correlation heatmap between car dynamics variables and different traffic contexts
    - each row is diff traffic conditions
        1：Area: suburban/urban
        2: Weather: rainy/sunny
        3: Traffic: flow/jam

    - each column is diff car dynamics variables
        Location x3
        Rotation x3
        Steering x1
        Throttle x1
        Brake x1
        Gear x1
        Linear_velocity x3
        Angular_velocity x3
        (Total: 16)

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
from scipy.stats import spearmanr
import seaborn as sns
import pickle
from scipy.stats import pointbiserialr

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
# generate 8x16 grid of subplots
# fig, axes = plt.subplots(8, 16)

# list task names & car dynamics
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

car_dynamics_name = [
    'location x',
    'location y',
    'location z',
    'rotation x',
    'rotation y',
    'rotation z',
    'steering',
    'throttle',
    'brake',
    'gear',
    'linear_velocity x',
    'linear_velocity y',
    'linear_velocity z',
    'angular_velocity x',
    'angular_velocity y',
    'angular_velocity z',
]

# generate dataframe to store final mean and variance values
Final_Mean = pd.DataFrame({('Task',''):range(1,9)})
Final_Var = pd.DataFrame({('Task',''):range(1,9)})


# for part_NO in range(1,51):
#     for car_dynamics in car_dynamics_name:
#         Final_Mean[(f'P{part_NO}', car_dynamics)] = None
#         Final_Var[(f'P{part_NO}', car_dynamics)] = None

# participant NO
for part_NO in range(1,51):
    folder_path = f'path\\to\\Clip_Car_Dynamics\\P{part_NO}\\'

    for filename in os.listdir(folder_path):

        match = re.match(r'(\d{4})_p\d+_(\d+)', filename)
        date = match.group(1)
        task = match.group(2)
        df = pd.read_excel(folder_path+filename)
        headers = df.columns.tolist()
        df = df.dropna()
        for header in headers:
            locals()[header] = df[header]

        # save diff car dynamics mean and variance
        # location x
        data = Location.apply(lambda x: ast.literal_eval(x)).apply(lambda x: x[0])
        Final_Mean.loc[int(task)-1,(f'P{part_NO}', car_dynamics_name[0])] = data.mean()
        Final_Var.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[0])] = data.var()
        # location y
        data = Location.apply(lambda x: ast.literal_eval(x)).apply(lambda x: x[1])
        Final_Mean.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[1])] = data.mean()
        Final_Var.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[1])] = data.var()
        # location z
        data = Location.apply(lambda x: ast.literal_eval(x)).apply(lambda x: x[2])
        Final_Mean.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[2])] = data.mean()
        Final_Var.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[2])] = data.var()

        # rotation x
        data = Rotation.apply(lambda x: ast.literal_eval(x)).apply(lambda x: x[0])
        Final_Mean.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[3])] = data.mean()
        Final_Var.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[3])] = data.var()
        # rotation y
        data = Rotation.apply(lambda x: ast.literal_eval(x)).apply(lambda x: x[1])
        Final_Mean.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[4])] = data.mean()
        Final_Var.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[4])] = data.var()
        # rotation z
        data = Rotation.apply(lambda x: ast.literal_eval(x)).apply(lambda x: x[2])
        Final_Mean.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[5])] = data.mean()
        Final_Var.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[5])] = data.var()

        # steering
        Final_Mean.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[6])] = Steering.mean()
        Final_Var.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[6])] = Steering.var()

        # throttle
        Final_Mean.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[7])] = Throttle.mean()
        Final_Var.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[7])] = Throttle.var()

        # brake
        Final_Mean.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[8])] = Brake.mean()
        Final_Var.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[8])] = Brake.var()

        # gear
        Final_Mean.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[9])] = Gear.mean()
        Final_Var.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[9])] = Gear.var()

        # Linear_velocity x
        data = linear_velocity.apply(lambda x: ast.literal_eval(x)).apply(lambda x: x[0])
        Final_Mean.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[10])] = data.mean()
        Final_Var.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[10])] = data.var()
        # Linear_velocity y
        data = linear_velocity.apply(lambda x: ast.literal_eval(x)).apply(lambda x: x[1])
        Final_Mean.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[11])] = data.mean()
        Final_Var.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[11])] = data.var()
        # Linear_velocity z
        data = linear_velocity.apply(lambda x: ast.literal_eval(x)).apply(lambda x: x[2])
        Final_Mean.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[12])] = data.mean()
        Final_Var.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[12])] = data.var()

        # Angular_velocity x
        data = angular_velocity.apply(lambda x: ast.literal_eval(x)).apply(lambda x: x[0])
        Final_Mean.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[13])] = data.mean()
        Final_Var.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[13])] = data.var()
        # Angular_velocity y
        data = angular_velocity.apply(lambda x: ast.literal_eval(x)).apply(lambda x: x[1])
        Final_Mean.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[14])] = data.mean()
        Final_Var.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[14])] = data.var()
        # Angular_velocity z
        data = angular_velocity.apply(lambda x: ast.literal_eval(x)).apply(lambda x: x[2])
        Final_Mean.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[15])] = data.mean()
        Final_Var.loc[int(task)-1, (f'P{part_NO}', car_dynamics_name[15])] = data.var()
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
for car_dynamic in car_dynamics_name:
    car_dynamic_mean = Final_Mean.xs(car_dynamic,level=1,axis=1)
    car_dynamic_mean = car_dynamic_mean.reset_index().melt(id_vars='index', var_name='Column', value_name='Value')
    data_mean = car_dynamic_mean['Value'].tolist()
    label_ind_mean = car_dynamic_mean['index'].tolist()

    car_dynamic_var = Final_Var.xs(car_dynamic, level=1, axis=1)
    car_dynamic_var = car_dynamic_var.reset_index().melt(id_vars='index', var_name='Column', value_name='Value')
    data_var = car_dynamic_var['Value'].tolist()
    label_ind_var = car_dynamic_var['index'].tolist()

    for category, value in traffic_contexts.items():
        label = [0 if value[0] in task_name[l] else 1 for ind, l in enumerate(label_ind_mean)]
        corr, p_value = pointbiserialr(data_mean, label)
        corr_map_mean.loc[category,car_dynamic] = corr

        label = [0 if value[0] in task_name[l] else 1 for ind, l in enumerate(label_ind_var)]
        corr, p_value = pointbiserialr(data_var, label)
        corr_map_var.loc[category, car_dynamic] = corr


plt.figure(1)
sns.heatmap(corr_map_mean, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Car Dynamics Mean')

plt.figure(2)
sns.heatmap(corr_map_var, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Car Dynamics Variance')

plt.show()


breakpoint()







