import pandas as pd

# Load files
submission = pd.read_csv(
    r'D:\MLproject semeval\results\submission.csv'
)

labels = pd.read_csv(
    r'D:\MLproject semeval\data\test_labels_subtask1.csv'
)

test_text = pd.read_csv(
    r'D:\MLproject semeval\data\test_subtask1.csv'
)

# Normalize labels
labels['valence'] = labels['valence'] + 2

print("\nTEST FILE COLUMNS:")
print(test_text.columns.tolist())

# Detect actual text column
text_col = None

for col in test_text.columns:

    if col.lower() in [
        'text',
        'sentence',
        'content',
        'utterance'
    ]:
        text_col = col

# If exact match not found
if text_col is None:

    for col in test_text.columns:

        if 'text' in col.lower():
            text_col = col
            break

print(f"\nUsing text column: {text_col}")

# Keep only needed columns
test_text = test_text[['text_id', text_col]]

# Merge
merged = submission.merge(labels, on='text_id')
merged = merged.merge(test_text, on='text_id')

# Calculate error
merged['error'] = abs(
    merged['valence'] - merged['pred_valence']
)

# Worst predictions
worst = merged.nlargest(5, 'error')

print("\n" + "="*70)
print("5 WORST PREDICTIONS")
print("="*70)

for _, row in worst.iterrows():

    print(f"\nText ID        : {row['text_id']}")
    print(f"True Valence   : {row['valence']:.1f}")
    print(f"Predicted      : {row['pred_valence']:.2f}")
    print(f"Absolute Error : {row['error']:.2f}")

    print("\nTEXT:")

    print(str(row.filter(like='text').iloc[-1])[:300])
    print("\n" + "-"*70)



  