import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download required NLTK resources
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

# Load cleaned dataset
df = pd.read_csv('../data/train_clean.csv')

# Convert timestamp column
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("Extracting features... please wait")

# FEATURE 1: Word Count
df['word_count'] = df['text'].apply(
    lambda x: len(str(x).split())
)
print("✓ Feature 1: word_count")

# FEATURE 2: Text Type
df['is_feeling_words'] = df['is_words'].astype(int)
print("✓ Feature 2: is_feeling_words")

# FEATURES 3,4,5: VADER Sentiment
sia = SentimentIntensityAnalyzer()

def get_vader(text):
    scores = sia.polarity_scores(str(text))
    return scores['compound'], scores['pos'], scores['neg']

vader = df['text'].apply(get_vader)

df['sentiment_polarity'] = vader.apply(lambda x: x[0])
df['positive_word_ratio'] = vader.apply(lambda x: x[1])
df['negative_word_ratio'] = vader.apply(lambda x: x[2])

print("✓ Features 3,4,5: VADER sentiment")

# FEATURE 6: Emotion Word Count
emotion_words = [
    'happy', 'sad', 'angry', 'anxious',
    'joy', 'fear', 'tired', 'excited',
    'calm', 'frustrated', 'content',
    'depressed', 'peaceful', 'nervous',
    'relaxed'
]

df['emotion_word_count'] = df['text'].apply(
    lambda x: sum(
        str(x).lower().split().count(w)
        for w in emotion_words
    )
)

print("✓ Feature 6: emotion_word_count")

# FEATURE 7: Average Word Length
df['avg_word_length'] = df['text'].apply(
    lambda x: np.mean(
        [len(w) for w in str(x).split()]
    ) if str(x).split() else 0
)

print("✓ Feature 7: avg_word_length")

# FEATURE 8: First-Person Pronouns
pronouns = ['i', 'me', 'my', 'myself', 'mine']

df['first_person_count'] = df['text'].apply(
    lambda x: sum(
        str(x).lower().split().count(p)
        for p in pronouns
    )
)

print("✓ Feature 8: first_person_count")

# FEATURE 9: Intensifiers
intensifiers = [
    'very', 'extremely', 'absolutely',
    'incredibly', 'totally',
    'really', 'so'
]

df['intensifier_count'] = df['text'].apply(
    lambda x: sum(
        str(x).lower().split().count(w)
        for w in intensifiers
    )
)

print("✓ Feature 9: intensifier_count")

# FEATURE 10: Exclamation Marks
df['exclamation_count'] = df['text'].apply(
    lambda x: str(x).count('!')
)

print("✓ Feature 10: exclamation_count")

# FEATURES 11,12: Time Features
df['hour_of_day'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

print("✓ Features 11,12: hour_of_day, day_of_week")

# FEATURE 13: Wave
print("✓ Feature 13: wave")

# FEATURE 14: Days Since First Entry
df['first_date'] = df.groupby('user_id')['timestamp'].transform('min')

df['days_since_first'] = (
    df['timestamp'] - df['first_date']
).dt.days

print("✓ Feature 14: days_since_first")

# Final feature list
feature_cols = [
    'word_count',
    'is_feeling_words',
    'sentiment_polarity',
    'positive_word_ratio',
    'negative_word_ratio',
    'emotion_word_count',
    'avg_word_length',
    'first_person_count',
    'intensifier_count',
    'exclamation_count',
    'hour_of_day',
    'day_of_week',
    'days_since_first'
]

print("\n=== FEATURE MATRIX PREVIEW ===")
print(df[feature_cols].head())

print("\nShape:")
print(df[feature_cols].shape)

# Save feature dataset
df.to_csv('../data/train_features.csv', index=False)

print("\n✅ ALL 14 FEATURES EXTRACTED!")
print("Saved to train_features.csv")