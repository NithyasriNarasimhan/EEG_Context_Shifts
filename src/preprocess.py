import scipy.io
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# Function to load EEG data and pad/truncate it as needed
def load_eeg_data(file_path, label):
    print(f"Loading data from {file_path}")
    mat_data = scipy.io.loadmat(file_path)
    key = list(mat_data.keys())[-1]  # assuming the last key holds the data
    eeg_data = mat_data[key]

    if isinstance(eeg_data, np.ndarray) and eeg_data.shape[0] == 1:
        eeg_data = eeg_data[0] 

    subjects = []
    for task in eeg_data:
        if isinstance(task, np.ndarray):
            for subject in task:
                subjects.append(subject.flatten()) 

    if not subjects:
        raise ValueError(f"No subjects found in {file_path}")

    subjects = np.array(subjects, dtype=object)
    max_len = max(map(len, subjects))  # find the longest vector
   
    subjects = np.array([np.pad(s, (0, max_len - len(s))) for s in subjects])

    return subjects, np.full(len(subjects), label)  # returns data and label array

# Main function to process all .mat files and save a dataset
def save_processed_data():
    files = {
        "MC": ("MC.mat", 0),
        "MADHD": ("MADHD.mat", 1),
        "FC": ("FC.mat", 0),
        "FADHD": ("FADHD.mat", 1)
    }

    X_list, y_list = [], []

    for name, (path, label) in files.items():
        try:
            X, y = load_eeg_data(path, label)
            X_list.append(X)
            y_list.append(y)
        except Exception as e:
            print(f"Oops! Couldn't load {name}: {e}")

   
    X = np.vstack(X_list)
    y = np.concatenate(y_list)

   
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df = pd.DataFrame(X_scaled)
    df["Label"] = y  # attach target column

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/processed_eeg_data.csv", index=False)
    print("Data is in processed_eeg_data.csv")

if __name__ == '__main__':
    save_processed_data()
