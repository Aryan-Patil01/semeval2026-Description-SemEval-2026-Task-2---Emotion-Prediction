import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

# Load updated dataset
df = pd.read_csv(r'D:\MLproject semeval\data\train_features_v2.csv')

# Updated feature list
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
    'days_since_first',
    'high_arousal_words',
    'low_arousal_words',
    'caps_ratio',
    'punct_intensity',
    'sentence_count',
    'sentiment_intensity'
]

# Prepare data
X = df[feature_cols].fillna(0)

X_tr, X_vl, yv_tr, yv_vl = train_test_split(
    X,
    df['valence'],
    test_size=0.2,
    random_state=42
)

_, _, ya_tr, ya_vl = train_test_split(
    X,
    df['arousal'],
    test_size=0.2,
    random_state=42
)

print("Training XGBoost with new arousal features...\n")

# Train models
xgb_v = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    verbosity=0
)

xgb_a = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    verbosity=0
)

xgb_v.fit(X_tr, yv_tr)
xgb_a.fit(X_tr, ya_tr)

# Predictions
pred_v = xgb_v.predict(X_vl)
pred_a = xgb_a.predict(X_vl)

# Scores
r_val, _ = pearsonr(yv_vl, pred_v)
r_aro, _ = pearsonr(ya_vl, pred_a)

print("=" * 50)
print("OLD TEST SCORES")
print("Valence : 0.572")
print("Arousal : 0.224")
print("=" * 50)

print("\nNEW VALIDATION SCORES")
print(f"Valence Pearson r : {r_val:.3f}")
print(f"Arousal Pearson r : {r_aro:.3f}")

print("\n✅ Retraining complete!")