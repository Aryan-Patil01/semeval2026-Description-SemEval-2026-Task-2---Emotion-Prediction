import pandas as pd
import numpy as np
import re
import nltk

from nltk.sentiment import SentimentIntensityAnalyzer
from xgboost import XGBRegressor

nltk.download('vader_lexicon', quiet=True)

# ==============================
# LOAD DATA
# ==============================

test = pd.read_csv(r'D:\MLproject semeval\data\test_subtask1.csv')
train = pd.read_csv(r'D:\MLproject semeval\data\train_features_v2.csv')

test['timestamp'] = pd.to_datetime(test['timestamp'])

sia = SentimentIntensityAnalyzer()

# ==============================
# FEATURE EXTRACTION
# ==============================

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

def extract_features(df):

    df = df.copy()

    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

    if 'is_words' in df.columns:
        df['is_feeling_words'] = df['is_words'].astype(int)
    else:
        df['is_feeling_words'] = 0

    vader = df['text'].apply(lambda x: sia.polarity_scores(str(x)))

    df['sentiment_polarity'] = vader.apply(lambda x: x['compound'])
    df['positive_word_ratio'] = vader.apply(lambda x: x['pos'])
    df['negative_word_ratio'] = vader.apply(lambda x: x['neg'])

    df['avg_word_length'] = df['text'].apply(
        lambda x: np.mean([len(w) for w in str(x).split()])
        if str(x).split() else 0
    )

    pronouns = ['i','me','my','myself','mine']

    df['first_person_count'] = df['text'].apply(
        lambda x: sum(str(x).lower().split().count(p) for p in pronouns)
    )

    intensifiers = ['very','extremely','absolutely','incredibly','really']

    df['intensifier_count'] = df['text'].apply(
        lambda x: sum(str(x).lower().split().count(w) for w in intensifiers)
    )

    df['exclamation_count'] = df['text'].apply(lambda x: str(x).count('!'))

    emotion_words = ['happy','sad','angry','tired','excited','calm','anxious']

    df['emotion_word_count'] = df['text'].apply(
        lambda x: sum(str(x).lower().split().count(w) for w in emotion_words)
    )

    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    if 'wave' not in df.columns:
        df['wave'] = 0

    df['first_date'] = df.groupby('user_id')['timestamp'].transform('min')
    df['days_since_first'] = (
        df['timestamp'] - df['first_date']
    ).dt.days

    # NEW AROUSAL FEATURES

    df['high_arousal_words'] = df['text'].apply(
        lambda x: sum(str(x).lower().split().count(w) for w in high_arousal)
    )

    df['caps_ratio'] = df['text'].apply(
        lambda x: sum(
            1 for w in str(x).split()
            if w.isupper() and len(w) > 1
        ) / max(len(str(x).split()), 1)
    )

    df['punct_intensity'] = df['text'].apply(
        lambda x: len(re.findall(r'[!?]{2,}', str(x)))
    )

    df['sentence_count'] = df['text'].apply(
        lambda x: len(re.split(r'[.!?]+', str(x)))
    )

    df['sentiment_intensity'] = df['text'].apply(
        lambda x: abs(sia.polarity_scores(str(x))['compound'])
    )

    return df

print("Extracting test features...")

test = extract_features(test)

# ==============================
# FEATURES
# ==============================

feature_cols = [
    'word_count','is_feeling_words','sentiment_polarity',
    'positive_word_ratio','negative_word_ratio','emotion_word_count',
    'avg_word_length','first_person_count','intensifier_count',
    'exclamation_count','hour_of_day','day_of_week',
    'days_since_first',
    'high_arousal_words','caps_ratio',
    'punct_intensity','sentence_count','sentiment_intensity'
]

X_train = train[feature_cols].fillna(0)
X_test  = test[feature_cols].fillna(0)

# ==============================
# TRAIN XGBOOST
# ==============================

print("Training XGBoost models...")

xgb_val = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    verbosity=0
)

xgb_aro = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    verbosity=0
)

xgb_val.fit(X_train, train['valence'])
xgb_aro.fit(X_train, train['arousal'])

# ==============================
# PREDICTIONS
# ==============================

print("Generating XGBoost predictions...")

test['pred_valence'] = xgb_val.predict(X_test)
test['pred_arousal'] = xgb_aro.predict(X_test)

submission = test[['text_id', 'pred_valence', 'pred_arousal']]

submission.to_csv(
    r'D:\MLproject semeval\results\xgb_submission.csv',
    index=False
)

print("\n✅ xgb_submission.csv created successfully!")
print(submission.head())