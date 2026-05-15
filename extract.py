import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import find_peaks

# 📁 UPDATE THIS PATH
DATASET_PATH = "WFDBRecords"
OUTPUT_CSV = "ecg_dataset.csv"

sampling_freq = 500  # from .hea file

rows = []

# ------------------------
# COUNT TOTAL FILES (for progress)
# ------------------------
total_files = sum(
    1 for root, _, files in os.walk(DATASET_PATH)
    for f in files if f.endswith(".mat")
)

count = 0


def extract_features(mat_path, hea_path, filename):
    try:
        # ------------------------
        # LOAD SIGNAL (.mat)
        # ------------------------
        mat = loadmat(mat_path)

        # FIX 1: robust key extraction
        if "val" in mat:
            data = mat["val"]
        else:
            key = [k for k in mat.keys() if not k.startswith("__")][0]
            data = mat[key]

        # shape handling (12,5000) → (5000,12)
        if data.shape[0] == 12:
            data = data.T

        # ------------------------
        # CHANNEL REDUCTION
        # ------------------------
        signal = np.mean(data, axis=1)

        # ------------------------
        # RAW FEATURES (IMPORTANT)
        # ------------------------
        mean_val = np.mean(signal)
        std_val  = np.std(signal)
        max_val  = np.max(signal)
        min_val  = np.min(signal)
        energy   = np.sum(signal ** 2)

        # ------------------------
        # ZERO CROSSING RATE
        # ------------------------
        zero_crossings = ((signal[:-1] * signal[1:]) < 0).sum()

        # ------------------------
        # NORMALIZE (for peaks only)
        # ------------------------
        if np.std(signal) != 0:
            signal = (signal - np.mean(signal)) / np.std(signal)

        # ------------------------
        # PEAK FEATURES
        # ------------------------
        peaks, _ = find_peaks(signal, distance=150, height=0.5)
        peak_count = len(peaks)

        # ------------------------
        # HEART RATE (BPM)
        # ------------------------
        duration = len(signal) / sampling_freq
        heart_rate = (peak_count / duration) * 60 if duration > 0 else 0

        # ------------------------
        # READ .hea FILE
        # ------------------------
        age = -1
        sex = -1
        label = "unknown"

        with open(hea_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            if line.startswith("#Age:"):
                age_str = line.split(":")[1].strip()
                age = int(age_str) if age_str.isdigit() else -1

            elif line.startswith("#Sex:"):
                sex_str = line.split(":")[1].strip()
                if sex_str == "Male":
                    sex = 1
                elif sex_str == "Female":
                    sex = 0
                else:
                    sex = -1

            elif line.startswith("#Dx:"):
                dx = line.split(":")[1].strip()
                label = dx.split(",")[0]  # take first diagnosis

        return {
            "filename": filename,
            "mean": mean_val,
            "std": std_val,
            "max": max_val,
            "min": min_val,
            "peak_count": peak_count,
            "heart_rate": heart_rate,
            "energy": energy,
            "zero_crossings": zero_crossings,
            "age": age,
            "sex": sex,
            "label": label
        }

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        return None


# ------------------------
# PROCESS DATASET
# ------------------------
for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".mat"):
            mat_path = os.path.join(root, file)
            hea_path = mat_path.replace(".mat", ".hea")

            if os.path.exists(hea_path):
                result = extract_features(mat_path, hea_path, file)

                if result:
                    rows.append(result)

            # ------------------------
            # PROGRESS PRINT (FIXED)
            # ------------------------
            count += 1
            if count % 100 == 0:
                print(f"Processed {count}/{total_files} files...")


# ------------------------
# SAVE DATASET
# ------------------------
df = pd.DataFrame(rows)

# ------------------------
# LABEL ENCODING + MAPPING
# ------------------------
label_codes, uniques = pd.factorize(df["label"])
df["label_encoded"] = label_codes

# Save label mapping
mapping = pd.DataFrame({
    "label": uniques,
    "code": range(len(uniques))
})
mapping.to_csv("label_mapping.csv", index=False)

# Save dataset
df.to_csv(OUTPUT_CSV, index=False)

print("\n✅ Dataset created:", OUTPUT_CSV)
print("📊 Total samples:", len(df))
print(df.head())