import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import re

nltk.download('vader_lexicon', quiet=True)

df = pd.read_csv(r'D:\MLproject semeval\data\train_features.csv')

sia = SentimentIntensityAnalyzer()

# ── FEATURE 15 & 16 ─────────────────────────────

high_arousal = [
    'excited','angry','furious','thrilled','terrified','desperate',
    'anxious','panicking','energetic','hyper','stressed','overwhelmed',
    'rushing','running','screaming','shocked','frantic','crazy',
    'exhausted','drained','pumped','motivated','restless'
]

low_arousal = [
    'calm','relaxed','peaceful','sleepy','tired','quiet',
    'bored','slow','lazy','content','still','gentle','mellow'
]

df['high_arousal_words'] = df['text'].apply(
    lambda x: sum(str(x).lower().split().count(w) for w in high_arousal)
)

df['low_arousal_words'] = df['text'].apply(
    lambda x: sum(str(x).lower().split().count(w) for w in low_arousal)
)

print("✓ Feature 15,16 added")

# ── FEATURE 17 ──────────────────────────────────

df['caps_ratio'] = df['text'].apply(
    lambda x: sum(
        1 for w in str(x).split()
        if w.isupper() and len(w) > 1
    ) / max(len(str(x).split()), 1)
)

print("✓ Feature 17 added")

# ── FEATURE 18 ──────────────────────────────────

df['punct_intensity'] = df['text'].apply(
    lambda x: len(re.findall(r'[!?]{2,}', str(x)))
)

print("✓ Feature 18 added")

# ── FEATURE 19 ──────────────────────────────────

df['sentence_count'] = df['text'].apply(
    lambda x: len(re.split(r'[.!?]+', str(x)))
)

print("✓ Feature 19 added")

# ── FEATURE 20 ──────────────────────────────────

df['sentiment_intensity'] = df['text'].apply(
    lambda x: abs(sia.polarity_scores(str(x))['compound'])
)

print("✓ Feature 20 added")

# Save new dataset
df.to_csv(
    r'D:\MLproject semeval\data\train_features_v2.csv',
    index=False
)

print("\n✅ Saved train_features_v2.csv successfully!")