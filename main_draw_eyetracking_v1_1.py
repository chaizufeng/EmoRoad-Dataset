"""
Description:
    This code aims to analyze eyetracking data by providing distribution of duration of gaze points on each screen.

Author:
    CHAI Bo & FANG Le (Leo)
"""
import numpy as np
import pandas as pd
import os
import zipfile
import re
import seaborn as sns
import matplotlib.pyplot as plt
from ref_timestamp import task_order



# store each task result
df_list = []
for task in range(1,9):
    df = pd.DataFrame(columns=['left', 'center', 'right'])
    df_list.append(df)

# iterate through each participant
for part_NO in range(1, 51):

    for task in range(1,9):
        folder_path = f"path\\to\\Clip_Eyetracking\\P{part_NO}\\Eyetracking\\"
        pattern = rf'.*{task}\.zip$'
        zip_files = os.listdir(folder_path)

        # find target zip file
        for zip_file in zip_files:
            match = re.match(pattern, zip_file)
            if match:
                zip_path = f"{folder_path}{zip_file}"
                # breakpoint()
                break

        # check if matched
        if not match:
            print(f"P{part_NO} missing task{task}")
            continue

        else:
            with zipfile.ZipFile(zip_path) as z:
                csv_left = [f for f in z.namelist() if f.endswith('_left_fixations.csv')][0]
                with z.open(csv_left) as f:
                    df = pd.read_csv(f)
                    total_fixation_duration = df[df['fixation detected on surface'] == True]['duration [ms]'].sum()
                    total_duration = df['duration [ms]'].sum()
                    ratio_left = total_fixation_duration / total_duration

                csv_center = [f for f in z.namelist() if f.endswith('_center_fixations.csv')][0]
                with z.open(csv_center) as f:
                    df = pd.read_csv(f)
                    total_fixation_duration = df[df['fixation detected on surface'] == True][
                        'duration [ms]'].sum()
                    total_duration = df['duration [ms]'].sum()
                    if total_duration == 0:
                        ratio_center = np.nan
                        print(f"P{part_NO} {csv_center} is not correct")

                    else:
                        ratio_center = total_fixation_duration / total_duration

                csv_right = [f for f in z.namelist() if f.endswith('_right_fixations.csv')][0]
                with z.open(csv_right) as f:
                    df = pd.read_csv(f)
                    total_fixation_duration = df[df['fixation detected on surface'] == True][
                        'duration [ms]'].sum()
                    total_duration = df['duration [ms]'].sum()
                    ratio_right = total_fixation_duration / total_duration


                new_row = pd.DataFrame({
                    'left': [ratio_left],
                    'center': [ratio_center],
                    'right': [ratio_right]
                })

                df_list[task_order[part_NO-1][task-1]-1] = pd.concat([df_list[task_order[part_NO-1][task-1]-1], new_row], ignore_index=True)



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

for task in range(1,9):
    plt.figure(figsize=(5, 5))
    sns.boxplot(data=df_list[task-1])
    plt.gca().set_xticklabels(['Left Screen', 'Middle Screen', 'Right Screen'])
    plt.title(task_name[task-1])
    plt.tight_layout()
    # plt.savefig(f"figures/{task_name[task - 1]}.png")
    plt.show()


breakpoint()


