import os
from scipy.io import loadmat
import numpy as np
import pandas as pd

# Path to your folder
folder_path = "WFDBRecords/01/010"

data_rows = []

# Label extraction
def get_label(hea_path):
    try:
        with open(hea_path, "r") as f:
            for line in f:
                if "#Dx" in line:
                    return line.split(":")[1].strip()
    except:
        return "Unknown"
    return "Unknown"


for file in os.listdir(folder_path):
    if file.endswith(".mat"):
        mat_path = os.path.join(folder_path, file)
        hea_path = os.path.join(folder_path, file.replace(".mat", ".hea"))

        try:
            mat_data = loadmat(mat_path)
            signal = mat_data.get("val")

            if signal is None:
                continue

            signal = signal.flatten()

            label = get_label(hea_path)

            features = {
                "mean": np.mean(signal),
                "std": np.std(signal),
                "max": np.max(signal),
                "min": np.min(signal),
                "label": label
            }

            data_rows.append(features)

        except Exception as e:
            print(f"Error: {file} → {e}")

# Create DataFrame
df = pd.DataFrame(data_rows)

# Convert labels to numbers
label_map = {
    "N": 0,
    "AF": 1,
    "AFL": 2,
    "PVC": 3,
    "Other": 4
}

df["label"] = df["label"].apply(lambda x: label_map.get(x, 4))

# Save CSV
df.to_csv("dataset.csv", index=False)

print("Dataset created!")
print(df.head())