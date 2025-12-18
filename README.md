# EmoRoad: A Multimodal Dataset of Psychological, Physiological, and Behavioral Responses in Diverse Driving Scenarios
**EmoRoad** is a multimodal dataset designed to fill the gap in drivers’ physiological and psychological behavioral data across diverse driving scenarios. By configuring multiple driving conditions, EmoRoad collects rich, synchronized data streams, including:

- **Physiological signals**: EEG  
- **Behavioral data**: eye-tracking, facial expressions, steering wheel touch information, vehicle dynamics, and screen recordings  
- **Psychological data**: emotion annotations  

The EmoRoad dataset is publicly available and can be accessed and downloaded via its DOI: 10.6084/m9.figshare.30773390.

## EmoRoad Dataset Organization and File Structure

### Dataset Structure

The EmoRoad dataset is organized into two main folders: **`Clip`** and **`RawData`**.

#### *`Clip` Folder*

The **`Clip`** folder contains segmented data for all valid experimental trials.  
These clips are generated according to the unified start and end timestamps defined in variables `start_tmsp_ns` and `end_tmsp_ns` in `ref_timestamp.py`. The main characteristics are:

- Each modality has been **cut according to the same experimental time intervals** recorded in `ref_timestamp.py`.
- All processed data are **stored in CSV format**.
- The **first column of every CSV file** contains the corresponding **Unix timestamp**, preserving temporal alignment across modalities.
- Due to storage limitations, the **raw facial video**, **raw eye-tracking data**, and **raw triple-screen recordings** are **not uploaded** as raw data; only their cliped data are available in the `Clip` folder.

#### *`RawData` Folder*

The **`RawData`** folder contains the **unprocessed, original recordings** for each modality (except the excluded large video files mentioned above). These files preserve the raw output as collected during the experiment, without clipping or preprocessing.

#### *Important Notes on Experimental Order*

To eliminate order effects, the **driving tasks were performed in a randomized sequence** during the experiment.  
However, for consistency, the data files are **stored following the fixed task order listed below** (except for eye-tracking):

- For **most modalities**, the storage order **matches the task indices (1–8)** in the table.
- For **eye-tracking**, the correspondence between **suffix numbers 1–8 in the EyeTracking file names** and the actual tasks is also recorded in the variable `task_order` in `ref_timestamp.py`.

#### *Driving Task Index Mapping*

| Task ID | Driving Task        | Task ID | Driving Task        |
|--------:|---------------------|--------:|---------------------|
| 1       | suburban-sunny-flow | 5       | urban-sunny-flow    |
| 2       | suburban-rainy-flow | 6       | urban-rainy-flow    |
| 3       | suburban-sunny-jam  | 7       | urban-sunny-jam     |
| 4       | suburban-rainy-jam  | 8       | urban-rainy-jam     |




## EmoRoad Codebase Introduction

All scripts in this repository were tested with MATLAB R2025b, EEGLAB v2025.1.0, and Python 3.12.


### `main_eeg_artifacts_remove_ica_v1_1.m`

This script uses **EEGLAB** to preprocess EEG data. The main steps include:

- Removing the baseline  
- Applying a **notch filter** to remove power-line noise at **50 Hz**  
- Applying a **band-pass filter** to retain EEG activity in the **0.5–40 Hz** range  
- Performing **ICA** and using **ICLabel** to remove artifacts  

ICLabel classifies each ICA component into seven categories:  
`[Brain, Muscle, Eye, Line Noise, Channel Noise, Heart, Other]`.  
For each component, ICLabel outputs the probability of belonging to each class. In this code, we use class-specific probability thresholds to decide whether a component is an artifact and should be rejected.

The current threshold vector is:

```matlab
thr = [NaN 0.6 0.6 0.7 0.7 0.7 NaN];   % [Brain Muscle Eye LineNoise ChanNoise Heart Other]
```
Note that the thresholds for **Brain** and **Other** are set to `NaN` to ensure that components whose top class is Brain or Other are not automatically rejected.
Researchers can adjust these thresholds according to the characteristics and quality of EEG data.

Using ICLabel greatly improves the efficiency of artifact removal and avoids manual inspection of each ICA component, making it well suited for large-scale EEG datasets.

### `main_eeg_artifacts_remove_large_amplitude_v1_1.m`

This script removes segments of EEG data with **abnormally large amplitudes**.  
In the current code, the amplitude threshold is set to **100 μV**.

Because EmoRoad is a **driving task**, participants inevitably move their heads (e.g., turning to check the rear-view mirrors), which can cause very large voltage fluctuations in the EEG signals. These large-amplitude artifacts are often **not fully removed by ICA**, so this script provides an additional amplitude-based rejection step to exclude such segments from further analysis.

### `main_corr_car_dynamics_v1_1.py`, `main_corr_eeg_v1_1.py`, `main_corr_emosense_v1_1.py`

These scripts perform correlation analyses between different signal modalities and three types of traffic context: **Weather**, **Traffic volume**, and **Road scenarios (Area)**.

- **`main_corr_car_dynamics_v1_1.py`**  
  Computes the **mean** and **variance** of car dynamics variables (e.g., speed, acceleration, steering) and analyzes their correlations with the three traffic context factors.

- **`main_corr_eeg_v1_1.py`**  
  Computes the **variance** of EEG-derived measures and evaluates how they correlate with Weather, Traffic volume, and Road scenarios.

- **`main_corr_emosense_v1_1.py`**  
  For each EmoSense spectrum, this script extracts:  
  - The **maximum amplitude**  
  - The **index (Frequency point) at which this maximum occurs**  

  It then computes the **mean** and **variance** of these two quantities over time and analyzes their correlations with the three traffic context factors (Weather, Traffic volume, Road scenarios).

### `main_draw_eyetracking_v1_1.py`

This script visualizes, for each driving task, the distribution of **fixation duration ratios** across the triple-screen setup.

For all participants and each task, it computes the proportion of total fixation time that falls on **each of the three screens**, and then plots the distribution of these **screen-wise duration ratios** across subjects.

### `main_draw_imotions_v1_1.py`

This script visualizes, for each driving task, the **average duration ratio** of each emotion category labeled by the participants.

In other words, for every emotion type and driving task, it computes the proportion of task time during which that emotion was present and plots the mean duration ratio across participants.

### `ref_timestamp.py`

As mentioned earlier, this script stores three key variables:

- `start_tmsp_ns` – the **start timestamp** (in nanoseconds) for each driving task of each participant  
- `end_tmsp_ns` – the **end timestamp** (in nanoseconds) for each driving task of each participant  
- `task_order` – the **order of driving tasks** for each participant  

Except for the **eye-tracking data**, all other modalities have already been **reordered** according to the task order shown in the **Driving Task Index Mapping** table using these timestamps and `task_order`.






