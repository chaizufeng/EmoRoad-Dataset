'''
Descriptions:
    This code aims to draw the average duration histogram of each emotion for each driving task.
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

    - each column is diff emotion labels
        'contempt', 'engagement', 'sentiment', 'confusion',
        'neutral', 'anger', 'disgust', 'fear',
        'joy', 'sadness', 'surprise'

Author:
    CHAI Bo & FANG Le (Leo)
'''

import os
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

# calculate duration of each emotion
def emo_duration(df):
    '''
    This func aims to calculate duration of each emotion
    df:
        data frame of each csv file
    return:
        duration of each emotion
    '''

    emotions = ['Neutral', 'Engagement', 'Joy', 'Fear',
                'Disgust', 'Confusion', 'Anger', 'Sentimentality',
                'Sadness', 'Surprise', 'Contempt']

    total_duration = df['Timestamp'].iloc[-1] - df['Timestamp'].iloc[0]

    duration = {emo: [] for emo in emotions}

    for emo in emotions:
        col = emo + ' instance on Neon Glasses'

        if emo != 'Neutral':
            if df[col].isna().all():
                duration[emo] = 0
                continue
            filtered_col = df[col].dropna()
            ranges = filtered_col.groupby(filtered_col).apply(
                lambda group: pd.Series({
                    'start': group.index.min(),
                    'end': group.index.max()
                })
            )
            filtered_df = df[['Timestamp', col]].dropna(subset=[col])
            time_diffs = (
                filtered_df.groupby(col)['Timestamp']
                .agg(['min', 'max'])
                .apply(lambda row: row['max'] - row['min'], axis=1)
            )
            total_time_diff = time_diffs.sum()
            duration[emo] = total_time_diff / total_duration

    # decide the duration of Neutral
    emo = 'Neutral'
    col = 'Neutral instance on Neon Glasses'
    columns_no_Neutral = [e + ' instance on Neon Glasses' for e in emotions if e != 'Neutral']
    nan_mask = df[columns_no_Neutral].isna().all(axis=1)   # mark NAN for all emotions simultaneously
    segment_starts = (nan_mask.diff() == 1) # mark nan_mask changes from True to False
    segment_ids = segment_starts.cumsum() + 1
    df[col] = nan_mask.where(nan_mask != False, np.nan).where(nan_mask == False, segment_ids)
    filtered_df = df[['Timestamp', col]].dropna(subset=[col])
    time_diffs = (
        filtered_df.groupby(col)['Timestamp']
        .agg(['min', 'max'])
        .apply(lambda row: row['max'] - row['min'], axis=1)
    )
    total_time_diff = time_diffs.sum()
    duration[emo] = total_time_diff / total_duration
            # breakpoint()
    duration = pd.Series(duration)
    # breakpoint()
    return duration


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

emotions = ['Neutral', 'Engagement', 'Joy', 'Fear',
         'Disgust', 'Confusion', 'Anger', 'Sentimentality',
        'Sadness', 'Surprise', 'Contempt' ]

# initialization
count = {emotion:[50]*8 for emotion in emotions}  # initial 50 means 50 participants
num_valid_emo_duration = pd.DataFrame(count)
dur = {emotion:[0]*8 for emotion in emotions}
part_emo_duration = pd.DataFrame(dur)
allpart_emo_duration = pd.DataFrame(dur)


# participant NO
for part_NO in range(1,51):
    folder_path = f'E:\\Clip_iMotions\\p{part_NO}\\'

    # iterate through each file
    for filename in os.listdir(folder_path):
        match = re.match(r'(\d{4})_p\d+_(\d+)', filename)
        date = match.group(1)
        task = match.group(2)

        # load imotions data
        try:
            df = pd.read_csv(folder_path+filename)
            if df.empty:
                duration = pd.Series({emotion:0 for emotion in emotions})
                num_valid_emo_duration.loc[int(task)-1] = num_valid_emo_duration.loc[int(task)-1] - 1
            else:
                duration = emo_duration(df)
            part_emo_duration.loc[int(task) - 1] = duration

            # breakpoint()
        except pd.errors.EmptyDataError:
            print(f'{filename} is empty!')
    allpart_emo_duration = allpart_emo_duration + part_emo_duration

mean_emo_duration = allpart_emo_duration / num_valid_emo_duration

# plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# set xy coordinates
xpos, ypos = np.meshgrid(np.arange(1, mean_emo_duration.shape[1]+1), np.arange(1, mean_emo_duration.shape[0]+1))
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos)

# set bar depth and width
dx = dy = 0.5
dz = mean_emo_duration.values.flatten()

# plot 3d bar chart
colors = plt.cm.tab10(np.linspace(0, 1, len(mean_emo_duration.columns)))  # color for diff emotions
for i, (x, y, z) in enumerate(zip(xpos, ypos, dz)):
    ax.bar3d(x, y, 0, dx, dy, z, color=colors[i % len(mean_emo_duration.columns)], alpha=0.8)

proxies = [plt.Rectangle((0, 0), 1, 1, fc=color) for color in colors]
ax.legend(proxies, emotions, loc='upper left', ncol=3)

# ax.set_xticks(np.arange(mean_emo_duration.shape[1]) + 0.5)
ax.set_xticklabels([])
# yticks = np.linspace(0, len(y) - 1, len(y))
ax.set_yticks(np.arange(1,mean_emo_duration.shape[0]+1) + 0.0)
ax.set_yticklabels(task_name, ha='left', rotation=-45)
ax.yaxis.set_tick_params(pad=1)
ax.set_zlabel('Value')
ax.set_title('3D Bar Plot of Emotion Distribution across Driving Tasks')
plt.tight_layout()
plt.show()

breakpoint()