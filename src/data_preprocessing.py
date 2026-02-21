from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

BASE_DIR = Path().resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "df_file.csv"

df = pd.read_csv(DATA_PATH)

# Remove ID column if exists
if "Id" in df.columns:
    df = df.drop(columns=["Id"])

print("Initial Shape:", df.shape)
print(df.head())
print(df.describe())

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

df["Text"] = df["Text"].apply(clean_text)

# Save Processed Data

processed_path = BASE_DIR / "data" / "processed" / "df_file.csv"
df.to_csv(processed_path, index=False)

print("Data preprocessing completed and saved.")