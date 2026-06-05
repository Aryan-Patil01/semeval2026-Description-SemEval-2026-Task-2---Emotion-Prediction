


import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import nltk, warnings
nltk.download('vader_lexicon', quiet=True)
warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_csv(r'D:\MLproject semeval\data\train_subtask2a.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['valence'] = df['valence'] + 2
df['arousal'] = df['arousal'] + 1
df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

# ── ORIGINAL 14 FEATURES ─────────────────────────────
print("Extracting text features...")
sia = SentimentIntensityAnalyzer()

df['word_count']          = df['text'].apply(lambda x: len(str(x).split()))
df['is_feeling_words']    = df['is_words'].astype(int)
vader = df['text'].apply(lambda x: sia.polarity_scores(str(x)))
df['sentiment_polarity']  = vader.apply(lambda x: x['compound'])
df['positive_word_ratio'] = vader.apply(lambda x: x['pos'])
df['negative_word_ratio'] = vader.apply(lambda x: x['neg'])
df['avg_word_length']     = df['text'].apply(
    lambda x: np.mean([len(w) for w in str(x).split()] or [0]))
pronouns = ['i','me','my','myself','mine']
df['first_person_count']  = df['text'].apply(
    lambda x: sum(str(x).lower().split().count(p) for p in pronouns))
df['exclamation_count']   = df['text'].apply(lambda x: str(x).count('!'))
intensifiers = ['very','extremely','absolutely','really','so']
df['intensifier_count']   = df['text'].apply(
    lambda x: sum(str(x).lower().split().count(w) for w in intensifiers))
df['hour_of_day']         = df['timestamp'].dt.hour
df['day_of_week']         = df['timestamp'].dt.dayofweek
df['collection_phase']    = df['collection_phase']
df['first_date']          = df.groupby('user_id')['timestamp'].transform('min')
df['days_since_first']    = (df['timestamp'] - df['first_date']).dt.days

# ── BONUS: State change features (unique to 2a!) ──────
df['state_change_valence'] = df['state_change_valence'].fillna(0)
df['state_change_arousal'] = df['state_change_arousal'].fillna(0)
print("✓ Text + state change features done")

# ── NEW: User history features ────────────────────────
print("Building user history features (takes ~2 mins)...")

user_mean_val, user_mean_aro = [], []
user_last_val, user_last_aro = [], []
user_val_std,  user_count    = [], []
user_val_trend               = []

for idx, row in df.iterrows():
    uid  = row['user_id']
    ts   = row['timestamp']
    past = df[(df['user_id'] == uid) & (df['timestamp'] < ts)]

    if len(past) == 0:
        user_mean_val.append(2.0);  user_mean_aro.append(1.0)
        user_last_val.append(2.0);  user_last_aro.append(1.0)
        user_val_std.append(0.0);   user_count.append(0)
        user_val_trend.append(0.0)
    else:
        user_mean_val.append(past['valence'].mean())
        user_mean_aro.append(past['arousal'].mean())
        user_last_val.append(past['valence'].iloc[-1])
        user_last_aro.append(past['arousal'].iloc[-1])
        user_val_std.append(past['valence'].std() if len(past)>1 else 0)
        user_count.append(len(past))
        if len(past) >= 2:
            mid = len(past) // 2
            trend = past['valence'].iloc[mid:].mean() - past['valence'].iloc[:mid].mean()
            user_val_trend.append(trend)
        else:
            user_val_trend.append(0.0)

df['user_mean_valence'] = user_mean_val
df['user_mean_arousal'] = user_mean_aro
df['user_last_valence']  = user_last_val
df['user_last_arousal']  = user_last_aro
df['user_val_std']       = user_val_std
df['user_entry_count']   = user_count
df['user_val_trend']     = user_val_trend
print("✓ User history features done")

# ── TRAIN MODELS ─────────────────────────────────────
feature_cols = [
    # Original text features
    'word_count', 'is_feeling_words', 'sentiment_polarity',
    'positive_word_ratio', 'negative_word_ratio', 'avg_word_length',
    'first_person_count', 'exclamation_count', 'intensifier_count',
    'hour_of_day', 'day_of_week', 'collection_phase', 'days_since_first',
    # Bonus state change features
    'state_change_valence', 'state_change_arousal',
    # New user history features
    'user_mean_valence', 'user_mean_arousal', 'user_last_valence',
    'user_last_arousal', 'user_val_std', 'user_entry_count', 'user_val_trend'
]

X = df[feature_cols].fillna(0)
y_val = df['valence']
y_aro = df['arousal']

X_tr, X_vl, yv_tr, yv_vl = train_test_split(X, y_val, test_size=0.2, random_state=42)
_,    _,    ya_tr, ya_vl  = train_test_split(X, y_aro, test_size=0.2, random_state=42)

print("\nTraining Random Forest...")
rf_v = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_a = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_v.fit(X_tr, yv_tr); rf_a.fit(X_tr, ya_tr)
r_v, _ = pearsonr(yv_vl, rf_v.predict(X_vl))
r_a, _ = pearsonr(ya_vl, rf_a.predict(X_vl))
print(f"  RF  → Valence r: {r_v:.3f}  Arousal r: {r_a:.3f}")

print("Training XGBoost...")
xgb_v = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbosity=0)
xgb_a = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbosity=0)
xgb_v.fit(X_tr, yv_tr); xgb_a.fit(X_tr, ya_tr)
r_v2, _ = pearsonr(yv_vl, xgb_v.predict(X_vl))
r_a2, _ = pearsonr(ya_vl, xgb_a.predict(X_vl))
print(f"  XGB → Valence r: {r_v2:.3f}  Arousal r: {r_a2:.3f}")

print("\n" + "="*45)
print("SUBTASK 2A RESULTS")
print("="*45)
print(f"Random Forest → Val: {r_v:.3f}  Aro: {r_a:.3f}")
print(f"XGBoost       → Val: {r_v2:.3f}  Aro: {r_a2:.3f}")
print("="*45)
print(f"\nTotal features used: {len(feature_cols)}")
print("\n✅ Screenshot this output and send to Claude!")

# Save features for next step
df.to_csv(r'D:\MLproject semeval\data\train_subtask2a_features.csv', index=False)
print("Saved train_subtask2a_features.csv")




# ── STEP 3: Score on official test labels ─────────────
print("\nLoading test labels for official scoring...")
labels2 = pd.read_csv(r'D:\MLproject semeval\data\test_labels_subtask2a_and_2b.csv')
print("Labels shape:", labels2.shape)
print("Labels columns:", labels2.columns.tolist())
print(labels2.head())

# Check what columns we have to work with
print("\nLabel value counts for subtask column (if exists):")
if 'subtask' in labels2.columns:
    print(labels2['subtask'].value_counts())

# Retrain on FULL training data (not split) for final prediction
print("\nRetraining on full training data...")
X_full = df[feature_cols].fillna(0)
xgb_v_full = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbosity=0)
xgb_a_full = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbosity=0)
xgb_v_full.fit(X_full, df['valence'])
xgb_a_full.fit(X_full, df['arousal'])
print("✓ Full retrain done")

# For each user in test set — get their training history and predict
test2 = pd.read_csv(r'D:\MLproject semeval\data\test_subtask2.csv')
print(f"\nTest users to predict: {len(test2)}")

predictions = []
for _, user_row in test2.iterrows():
    uid = user_row['user_id']
    user_hist = df[df['user_id'] == uid]

    if len(user_hist) == 0:
        # User not in training — use global mean
        predictions.append({
            'user_id': uid,
            'pred_valence': df['valence'].mean(),
            'pred_arousal': df['arousal'].mean()
        })
    else:
        # Use last entry features to predict next state
        last = user_hist.iloc[[-1]][feature_cols].fillna(0)
        # Update history features to reflect full history
        last['user_mean_valence'] = user_hist['valence'].mean()
        last['user_mean_arousal'] = user_hist['arousal'].mean()
        last['user_last_valence']  = user_hist['valence'].iloc[-1]
        last['user_last_arousal']  = user_hist['arousal'].iloc[-1]
        last['user_entry_count']   = len(user_hist)
        last['user_val_std']       = user_hist['valence'].std() if len(user_hist)>1 else 0

        pred_v = xgb_v_full.predict(last)[0]
        pred_a = xgb_a_full.predict(last)[0]
        predictions.append({
            'user_id': uid,
            'pred_valence': pred_v,
            'pred_arousal': pred_a
        })

pred_df = pd.DataFrame(predictions)
print("\nPredictions made:", len(pred_df))
print(pred_df.head())




# ── FIXED SCORING: Subtask 2a predicts CHANGE, not absolute ──
print("\nFixed scoring for Subtask 2a...")

labels2 = pd.read_csv(r'D:\MLproject semeval\data\test_labels_subtask2a_and_2b.csv')
print("Label columns:", labels2.columns.tolist())

# For each test user — compute their LAST known valence/arousal from training
# Then compute predicted CHANGE vs actual change
test2 = pd.read_csv(r'D:\MLproject semeval\data\test_subtask2.csv')

pred_changes_val = []
pred_changes_aro = []
true_changes_val = []
true_changes_aro = []

for _, user_row in test2.iterrows():
    uid = user_row['user_id']
    user_hist = df[df['user_id'] == uid].sort_values('timestamp')

    if len(user_hist) == 0:
        continue

    # Get label row for this user
    user_label = labels2[labels2['user_id'] == uid]
    if len(user_label) == 0:
        continue

    # Build features from user history
    last = user_hist.iloc[[-1]][feature_cols].fillna(0)
    last['user_mean_valence'] = user_hist['valence'].mean()
    last['user_mean_arousal'] = user_hist['arousal'].mean()
    last['user_last_valence']  = user_hist['valence'].iloc[-1]
    last['user_last_arousal']  = user_hist['arousal'].iloc[-1]
    last['user_entry_count']   = len(user_hist)
    last['user_val_std']       = user_hist['valence'].std() if len(user_hist)>1 else 0

    # Predict next absolute valence/arousal
    pred_v = xgb_v_full.predict(last)[0]
    pred_a = xgb_a_full.predict(last)[0]

    # Compute PREDICTED change = predicted next - last known
    last_val = user_hist['valence'].iloc[-1]
    last_aro = user_hist['arousal'].iloc[-1]
    pred_changes_val.append(pred_v - last_val)
    pred_changes_aro.append(pred_a - last_aro)

    # True change from label file
    true_changes_val.append(user_label['disp_change_valence'].values[0])
    true_changes_aro.append(user_label['disp_change_arousal'].values[0])

print(f"Users scored: {len(pred_changes_val)}")

if len(pred_changes_val) > 1:
    r_v, _ = pearsonr(true_changes_val, pred_changes_val)
    r_a, _ = pearsonr(true_changes_aro, pred_changes_aro)

    print("\n" + "="*50)
    print("🎯 SUBTASK 2A OFFICIAL TEST SCORES")
    print("="*50)
    print(f"Valence change Pearson r : {r_v:.3f}")
    print(f"Arousal change Pearson r : {r_a:.3f}")
    print("="*50)
    print("\nCompare with Subtask 1: Val=0.669  Aro=0.504")
    print("✅ Screenshot and send to Claude!")
else:
    print("Not enough matched users — paste output to Claude!")

# Save predictions
pred_out = pd.DataFrame({
    'user_id': test2['user_id'],
    'pred_change_valence': pred_changes_val + [0]*(len(test2)-len(pred_changes_val)),
    'pred_change_arousal': pred_changes_aro + [0]*(len(test2)-len(pred_changes_aro))
})
pred_out.to_csv(r'D:\MLproject semeval\results\submission_subtask2a.csv', index=False)
print("Saved submission_subtask2a.csv")
