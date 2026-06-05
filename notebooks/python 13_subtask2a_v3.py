import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GroupKFold
from scipy.stats import pearsonr
import nltk, warnings
nltk.download('vader_lexicon', quiet=True)
warnings.filterwarnings('ignore')

# ── LOAD ──────────────────────────────────────────
print("Loading Subtask 2a data...")
df = pd.read_csv(r'D:\MLproject semeval\data\train_subtask2a.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['valence'] = df['valence'] + 2
df['arousal'] = df['arousal'] + 1
df['state_change_valence'] = df['state_change_valence'].fillna(0)
df['state_change_arousal'] = df['state_change_arousal'].fillna(0)
df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

# ── TEXT FEATURES ─────────────────────────────────
sia = SentimentIntensityAnalyzer()
df['word_count']         = df['text'].apply(lambda x: len(str(x).split()))
df['char_count']         = df['text'].apply(lambda x: len(str(x)))
df['is_feeling_words']   = df['is_words'].astype(int)
vader = df['text'].apply(lambda x: sia.polarity_scores(str(x)))
df['sentiment_polarity']  = vader.apply(lambda x: x['compound'])
df['positive_word_ratio'] = vader.apply(lambda x: x['pos'])
df['negative_word_ratio'] = vader.apply(lambda x: x['neg'])
df['neutral_word_ratio']  = vader.apply(lambda x: x['neu'])
df['avg_word_length']     = df['text'].apply(
    lambda x: np.mean([len(w) for w in str(x).split()] or [0]))
pronouns = ['i', 'me', 'my', 'myself']
df['first_person_count']  = df['text'].apply(
    lambda x: sum(str(x).lower().split().count(p) for p in pronouns))
df['exclamation_count']   = df['text'].apply(lambda x: str(x).count('!'))
df['question_count']      = df['text'].apply(lambda x: str(x).count('?'))
df['ellipsis_count']      = df['text'].apply(lambda x: str(x).count('...'))
df['caps_ratio']          = df['text'].apply(
    lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1))

# ── TIME FEATURES ─────────────────────────────────
df['hour_of_day']      = df['timestamp'].dt.hour
df['day_of_week']      = df['timestamp'].dt.dayofweek
df['is_weekend']       = (df['day_of_week'] >= 5).astype(int)
df['is_night']         = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype(int)
df['collection_phase'] = df['collection_phase']
df['first_date']       = df.groupby('user_id')['timestamp'].transform('min')
df['days_since_first'] = (df['timestamp'] - df['first_date']).dt.days

# ── USER HISTORY FEATURES (no leakage) ────────────
print("Building user history features...")
records = {
    'user_mean_valence': [], 'user_mean_arousal': [],
    'user_last_valence': [], 'user_last_arousal': [],
    'user_val_std': [],      'user_aro_std': [],
    'user_entry_count': [],  'user_val_trend': [],
    'user_aro_trend': [],    'user_last2_val_delta': [],
    'user_last2_aro_delta': []
}

for idx, row in df.iterrows():
    uid  = row['user_id']
    ts   = row['timestamp']
    past = df[(df['user_id'] == uid) & (df['timestamp'] < ts)]

    if len(past) == 0:
        records['user_mean_valence'].append(2.0)
        records['user_mean_arousal'].append(1.0)
        records['user_last_valence'].append(2.0)
        records['user_last_arousal'].append(1.0)
        records['user_val_std'].append(0.0)
        records['user_aro_std'].append(0.0)
        records['user_entry_count'].append(0)
        records['user_val_trend'].append(0.0)
        records['user_aro_trend'].append(0.0)
        records['user_last2_val_delta'].append(0.0)
        records['user_last2_aro_delta'].append(0.0)
    else:
        pv = past['valence']
        pa = past['arousal']
        records['user_mean_valence'].append(pv.mean())
        records['user_mean_arousal'].append(pa.mean())
        records['user_last_valence'].append(pv.iloc[-1])
        records['user_last_arousal'].append(pa.iloc[-1])
        records['user_val_std'].append(pv.std() if len(past) > 1 else 0.0)
        records['user_aro_std'].append(pa.std() if len(past) > 1 else 0.0)
        records['user_entry_count'].append(len(past))
        if len(past) >= 2:
            mid = len(past) // 2
            records['user_val_trend'].append(pv.iloc[mid:].mean() - pv.iloc[:mid].mean())
            records['user_aro_trend'].append(pa.iloc[mid:].mean() - pa.iloc[:mid].mean())
            records['user_last2_val_delta'].append(pv.iloc[-1] - pv.iloc[-2])
            records['user_last2_aro_delta'].append(pa.iloc[-1] - pa.iloc[-2])
        else:
            records['user_val_trend'].append(0.0)
            records['user_aro_trend'].append(0.0)
            records['user_last2_val_delta'].append(0.0)
            records['user_last2_aro_delta'].append(0.0)

for col, vals in records.items():
    df[col] = vals

print("✓ Features done")

# ── DELTA FEATURES: current text sentiment vs user baseline ──
df['sentiment_vs_baseline'] = df['sentiment_polarity'] - df['user_mean_valence'].map(
    lambda x: (x - 2) / 2)  # rough normalisation
df['valence_deviation']     = df['user_last_valence'] - df['user_mean_valence']
df['arousal_deviation']     = df['user_last_arousal'] - df['user_mean_arousal']

feature_cols = [
    # text
    'word_count', 'char_count', 'is_feeling_words',
    'sentiment_polarity', 'positive_word_ratio', 'negative_word_ratio', 'neutral_word_ratio',
    'avg_word_length', 'first_person_count', 'exclamation_count',
    'question_count', 'ellipsis_count', 'caps_ratio',
    # time
    'hour_of_day', 'day_of_week', 'is_weekend', 'is_night',
    'collection_phase', 'days_since_first',
    # user history (NO targets!)
    'user_mean_valence', 'user_mean_arousal',
    'user_last_valence', 'user_last_arousal',
    'user_val_std', 'user_aro_std',
    'user_entry_count', 'user_val_trend', 'user_aro_trend',
    'user_last2_val_delta', 'user_last2_aro_delta',
    # delta
    'sentiment_vs_baseline', 'valence_deviation', 'arousal_deviation',
]

# ── TRAINING DATA ──────────────────────────────────
df_change = df[df['user_entry_count'] > 0].copy()
y_val = df_change['state_change_valence']
y_aro = df_change['state_change_arousal']
X     = df_change[feature_cols].fillna(0)
groups = df_change['user_id']

# ── USER-LEVEL CROSS-VALIDATION (no user leakage) ─
print("\nRunning GroupKFold CV...")
gkf = GroupKFold(n_splits=5)

def make_model():
    return XGBRegressor(
        n_estimators=500, learning_rate=0.02, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0
    )

cv_rv, cv_ra = [], []
for train_idx, val_idx in gkf.split(X, y_val, groups):
    Xtr, Xvl = X.iloc[train_idx], X.iloc[val_idx]
    yv_tr, yv_vl = y_val.iloc[train_idx], y_val.iloc[val_idx]
    ya_tr, ya_vl = y_aro.iloc[train_idx], y_aro.iloc[val_idx]
    mv, ma = make_model(), make_model()
    mv.fit(Xtr, yv_tr); ma.fit(Xtr, ya_tr)
    cv_rv.append(pearsonr(yv_vl, mv.predict(Xvl))[0])
    cv_ra.append(pearsonr(ya_vl, ma.predict(Xvl))[0])

print(f"CV Valence r : {np.mean(cv_rv):.3f} ± {np.std(cv_rv):.3f}")
print(f"CV Arousal r : {np.mean(cv_ra):.3f} ± {np.std(cv_ra):.3f}")

# ── RETRAIN ON FULL TRAINING DATA ─────────────────
print("\nRetraining on full data...")
xgb_v = make_model(); xgb_v.fit(X, y_val)
xgb_a = make_model(); xgb_a.fit(X, y_aro)

# ── TEST INFERENCE ────────────────────────────────
labels2 = pd.read_csv(r'D:\MLproject semeval\data\test_labels_subtask2a_and_2b.csv')
test2   = pd.read_csv(r'D:\MLproject semeval\data\test_subtask2.csv')

pred_cv, pred_ca, true_cv, true_ca = [], [], [], []

for _, urow in test2.iterrows():
    uid       = urow['user_id']
    user_hist = df[df['user_id'] == uid].sort_values('timestamp')
    ulbl      = labels2[labels2['user_id'] == uid]
    if len(user_hist) == 0 or len(ulbl) == 0:
        continue

    last = user_hist.iloc[[-1]][feature_cols].fillna(0).copy()

    # Recompute history features for the test snapshot
    pv = user_hist['valence']
    pa = user_hist['arousal']
    last['user_mean_valence']    = pv.mean()
    last['user_mean_arousal']    = pa.mean()
    last['user_last_valence']    = pv.iloc[-1]
    last['user_last_arousal']    = pa.iloc[-1]
    last['user_val_std']         = pv.std() if len(pv) > 1 else 0
    last['user_aro_std']         = pa.std() if len(pa) > 1 else 0
    last['user_entry_count']     = len(user_hist)
    last['valence_deviation']    = pv.iloc[-1] - pv.mean()
    last['arousal_deviation']    = pa.iloc[-1] - pa.mean()
    if len(pv) >= 2:
        mid = len(pv) // 2
        last['user_val_trend']       = pv.iloc[mid:].mean() - pv.iloc[:mid].mean()
        last['user_aro_trend']       = pa.iloc[mid:].mean() - pa.iloc[:mid].mean()
        last['user_last2_val_delta'] = pv.iloc[-1] - pv.iloc[-2]
        last['user_last2_aro_delta'] = pa.iloc[-1] - pa.iloc[-2]
    else:
        last['user_val_trend'] = last['user_aro_trend'] = 0
        last['user_last2_val_delta'] = last['user_last2_aro_delta'] = 0

    pred_cv.append(xgb_v.predict(last)[0])
    pred_ca.append(xgb_a.predict(last)[0])
    true_cv.append(ulbl['disp_change_valence'].values[0])
    true_ca.append(ulbl['disp_change_arousal'].values[0])

def safe_pearsonr(y_true, y_pred):
    if len(y_true) < 2 or np.std(y_pred) == 0 or np.std(y_true) == 0:
        return float('nan')
    return pearsonr(y_true, y_pred)[0]

r_v_test = safe_pearsonr(true_cv, pred_cv)
r_a_test = safe_pearsonr(true_ca, pred_ca)

print("\n" + "="*50)
print("🎯 SUBTASK 2A v3 OFFICIAL SCORES")
print("="*50)
print(f"Valence change r : {r_v_test:.3f}  (prev: -0.081)")
print(f"Arousal change r : {r_a_test:.3f}  (prev:  0.262)")
print("="*50)
print("✅ Screenshot and send to Claude!")

pd.DataFrame({
    'user_id':         test2['user_id'][:len(pred_cv)],
    'pred_change_val': pred_cv,
    'pred_change_aro': pred_ca
}).to_csv(r'D:\MLproject semeval\results\submission_subtask2a_v3.csv', index=False)
print("Saved submission_subtask2a_v3.csv")