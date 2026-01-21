# debug_data.py
import numpy as np
import pandas as pd

data = np.load("experiment_framework/data/processed/wikipedia/ml_wikipedia.npy")
print("Shape:", data.shape)
print(f"Data shape: {data.shape}")
print(f"Data sample: {data[0]}")
print("Min:", data.min(), "Max:", data.max())
print("Are values integers?", np.all(data == data.astype(int)))
print("Sample edges:", data[:3, :2])
if data.shape[1] > 2:
    print("Sample features:", data[:3, 2:5])

csv_path =  "experiment_framework/data/processed/wikipedia/ml_wikipedia.csv"
df = pd.read_csv(csv_path)
print("dataframe")
print(df.head())
print()
timestamps = np.genfromtxt(
    csv_path, 
    delimiter=",",
    skip_header=1,
    dtype=np.float64,
    filling_values=0.0)[:, 2].astype(np.float32)
    
print("timestamps shape",timestamps.shape)