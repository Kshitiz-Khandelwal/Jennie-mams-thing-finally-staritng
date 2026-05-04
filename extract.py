import os
from scipy.io import loadmat
import numpy as np
import pandas as pd

# Path to your folder
folder_path = "WFDBRecords/01/010"

data_rows = []

for file in os.listdir(folder_path):
    if file.endswith(".mat"):
        file_path = os.path.join(folder_path, file)

        try:
            mat_data = loadmat(file_path)

            # Most ECG data stored under 'val'
            signal = mat_data.get('val')

            if signal is None:
                continue

            signal = signal.flatten()

            # Basic features
            features = {
                "file": file,
                "mean": np.mean(signal),
                "std": np.std(signal),
                "max": np.max(signal),
                "min": np.min(signal)
            }

            data_rows.append(features)

        except Exception as e:
            print(f"Error with {file}: {e}")

# Convert to DataFrame
df = pd.DataFrame(data_rows)

# Save as CSV
df.to_csv("dataset_sample.csv", index=False)

print(df.head())