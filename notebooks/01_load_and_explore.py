import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# STEP 1: Load the data
df = pd.read_csv('../data/train_subtask1.csv')

print("=== FIRST 5 ROWS ===")
print(df.head())

print("\n=== COLUMN NAMES ===")
print(df.columns.tolist())

print("\n=== SHAPE (rows, columns) ===")
print(df.shape)

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# IMPORTANT:
# Check whether valence/arousal columns exist
print("\n=== DATA PREVIEW ===")
print(df.head())

# STEP 2 (only if columns exist)
if 'valence' in df.columns and 'arousal' in df.columns:

    print("\n=== VALENCE & AROUSAL STATS (RAW) ===")
    print(df[['valence', 'arousal']].describe())

    # Normalize scale
    df['valence'] = df['valence'] + 2
    df['arousal'] = df['arousal'] + 1

    print("\n=== VALENCE & AROUSAL STATS (NORMALIZED) ===")
    print(df[['valence', 'arousal']].describe())

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.hist(df['valence'], bins=20)
    plt.title('Valence Distribution')

    plt.subplot(1,2,2)
    plt.hist(df['arousal'], bins=10)
    plt.title('Arousal Distribution')

    plt.tight_layout()

    plt.savefig('../results/01_distributions.png')

    plt.show()

    print("Chart saved!")

else:
    print("\nNo valence/arousal columns found.")
    print("Check dataset structure first.")

# STEP 3: Save cleaned data
df.to_csv('../data/train_clean.csv', index=False)

print("Clean data saved!")
