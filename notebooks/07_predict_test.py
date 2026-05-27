import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon', quiet=True)

# ================================
# LOAD TEST DATA
# ================================

test = pd.read_csv(
    r'D:\MLproject semeval\data\test_subtask1.csv'
)

test['timestamp'] = pd.to_datetime(test['timestamp'])

# ================================
# FEATURE EXTRACTION
# ================================

sia = SentimentIntensityAnalyzer()

def extract_features(df):

    df = df.copy()

    # Feature 1
    df['word_count'] = df['text'].apply(
        lambda x: len(str(x).split())
    )

    # Feature 2
    df['is_feeling_words'] = df['is_words'].astype(int)

    # Features 3,4,5
    vader = df['text'].apply(
        lambda x: sia.polarity_scores(str(x))
    )

    df['sentiment_polarity'] = vader.apply(
        lambda x: x['compound']
    )

    df['positive_word_ratio'] = vader.apply(
        lambda x: x['pos']
    )

    df['negative_word_ratio'] = vader.apply(
        lambda x: x['neg']
    )

    # Feature 6
    emotion_words = [
        'happy',
        'sad',
        'angry',
        'tired',
        'excited',
        'calm',
        'anxious'
    ]

    df['emotion_word_count'] = df['text'].apply(
        lambda x: sum(
            str(x).lower().split().count(w)
            for w in emotion_words
        )
    )

    # Feature 7
    df['avg_word_length'] = df['text'].apply(
        lambda x: np.mean(
            [len(w) for w in str(x).split()]
        ) if str(x).split() else 0
    )

    # Feature 8
    pronouns = ['i', 'me', 'my', 'myself', 'mine']

    df['first_person_count'] = df['text'].apply(
        lambda x: sum(
            str(x).lower().split().count(p)
            for p in pronouns
        )
    )

    # Feature 9
    intensifiers = [
        'very',
        'extremely',
        'absolutely',
        'incredibly',
        'really'
    ]

    df['intensifier_count'] = df['text'].apply(
        lambda x: sum(
            str(x).lower().split().count(w)
            for w in intensifiers
        )
    )

    # Feature 10
    df['exclamation_count'] = df['text'].apply(
        lambda x: str(x).count('!')
    )

    # Features 11,12
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # Feature 14
    df['first_date'] = df.groupby(
        'user_id'
    )['timestamp'].transform('min')

    df['days_since_first'] = (
        df['timestamp'] - df['first_date']
    ).dt.days

    return df


print("Extracting test features...")

test = extract_features(test)

# ================================
# FEATURE LIST
# ================================

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

X_test = test[feature_cols].fillna(0)

# ================================
# LOAD TRAINING FEATURES
# ================================

train = pd.read_csv(
    r'D:\MLproject semeval\data\train_features.csv'
)

X_train = train[feature_cols].fillna(0)

# ================================
# TRAIN MODELS
# ================================

from sklearn.ensemble import RandomForestRegressor

print("Training Random Forest models...")

rf_val = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_aro = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_val.fit(X_train, train['valence'])
rf_aro.fit(X_train, train['arousal'])

# ================================
# GENERATE PREDICTIONS
# ================================

print("Generating predictions...")

test['pred_valence'] = rf_val.predict(X_test)
test['pred_arousal'] = rf_aro.predict(X_test)

# ================================
# CREATE SUBMISSION FILE
# ================================

submission = test[
    ['text_id', 'pred_valence', 'pred_arousal']
]

submission.to_csv(r'D:\MLproject semeval\results\rf_submission.csv', index=False)

# ================================
# FINAL OUTPUT
# ================================

print("\n✅ Submission file created successfully!")
print("\nPreview:\n")

print(submission.head())

print("\nSaved to:")
print(r'D:\MLproject semeval\results\submission.csv')