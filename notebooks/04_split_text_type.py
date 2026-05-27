import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'D:\MLproject semeval\data\train_features.csv')

feature_cols = [
    'word_count',
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

essays = df[df['is_feeling_words'] == 0]
fwords = df[df['is_feeling_words'] == 1]

results = {}

for name, subset in [('Essays', essays), ('FeelingWords', fwords)]:

    X = subset[feature_cols].fillna(0)
    y = subset['valence']

    X_tr, X_vl, y_tr, y_vl = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_tr, y_tr)

    pred = rf.predict(X_vl)

    r, _ = pearsonr(y_vl, pred)

    results[name] = r

    print(f"{name} → Valence Pearson r: {r:.3f}")

print("\nDone! Compare with original model.")