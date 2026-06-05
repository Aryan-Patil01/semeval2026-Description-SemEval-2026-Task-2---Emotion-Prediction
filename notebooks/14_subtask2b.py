
import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import nltk, warnings
nltk.download('vader_lexicon', quiet=True)
warnings.filterwarnings('ignore')

print("Loading Subtask 2b data...")
df = pd.read_csv(r'D:\MLproject semeval\data\train_subtask2b.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['valence'] = df['valence'] + 2
df['arousal'] = df['arousal'] + 1
df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

# Use 'group' as the wave column (subtask2b uses 'group' not 'collection_phase')
wave_col = 'group' if 'group' in df.columns else df.columns[4]
print(f"Using wave column: '{wave_col}'")
print("Shape:", df.shape, "Columns:", df.columns.tolist())

# ── TEXT FEATURES ──────────────────────────────────────
sia = SentimentIntensityAnalyzer()
df['word_count']        = df['text'].apply(lambda x: len(str(x).split()))
vader = df['text'].apply(lambda x: sia.polarity_scores(str(x)))
df['sentiment_polarity'] = vader.apply(lambda x: x['compound'])
df['positive_ratio']    = vader.apply(lambda x: x['pos'])
df['negative_ratio']    = vader.apply(lambda x: x['neg'])
pronouns = ['i', 'me', 'my', 'myself']
df['first_person']      = df['text'].apply(
    lambda x: sum(str(x).lower().split().count(p) for p in pronouns))
print("✓ Text features done")

# ── WAVE-LEVEL AGGREGATION using 'group' column ────────
print("Aggregating by user + wave...")
wave_features = df.groupby(['user_id', wave_col]).agg(
    mean_valence   =('valence',           'mean'),
    mean_arousal   =('arousal',           'mean'),
    std_valence    =('valence',           'std'),
    std_arousal    =('arousal',           'std'),
    min_valence    =('valence',           'min'),
    max_valence    =('valence',           'max'),
    mean_sentiment =('sentiment_polarity', 'mean'),
    mean_pos_ratio =('positive_ratio',    'mean'),
    mean_neg_ratio =('negative_ratio',    'mean'),
    mean_wc        =('word_count',        'mean'),
    mean_fp        =('first_person',      'mean'),
    text_count     =('text',              'count')
).reset_index()
wave_features.fillna(0, inplace=True)
print("Wave features shape:", wave_features.shape)

# ── BUILD CROSS-WAVE DATASET ───────────────────────────
print("Building cross-wave dataset...")
rows = []
for uid in wave_features['user_id'].unique():
    uw = wave_features[wave_features['user_id']==uid].sort_values(wave_col)
    if len(uw) < 2:
        continue
    for i in range(len(uw) - 1):
        curr = uw.iloc[i]
        nxt  = uw.iloc[i+1]
        pw   = uw.iloc[:i+1]
        rows.append({
            'user_id':         uid,
            'curr_wave':        curr[wave_col],
            'curr_mean_val':    curr['mean_valence'],
            'curr_mean_aro':    curr['mean_arousal'],
            'curr_std_val':     curr['std_valence'],
            'curr_std_aro':     curr['std_arousal'],
            'curr_min_val':     curr['min_valence'],
            'curr_max_val':     curr['max_valence'],
            'curr_sentiment':   curr['mean_sentiment'],
            'curr_pos':         curr['mean_pos_ratio'],
            'curr_neg':         curr['mean_neg_ratio'],
            'curr_wc':          curr['mean_wc'],
            'curr_fp':          curr['mean_fp'],
            'curr_tc':          curr['text_count'],
            'hist_mean_val':    pw['mean_valence'].mean(),
            'hist_mean_aro':    pw['mean_arousal'].mean(),
            'hist_val_trend':   pw['mean_valence'].diff().mean(),
            'hist_wave_count':  len(pw),
            'val_target':       nxt['mean_valence'] - curr['mean_valence'],
            'aro_target':       nxt['mean_arousal']  - curr['mean_arousal'],
        })

cross_df = pd.DataFrame(rows).fillna(0)
print(f"Cross-wave rows: {len(cross_df)} from {cross_df['user_id'].nunique()} users")

feat_cols = [c for c in cross_df.columns
             if c not in ['user_id', 'val_target', 'aro_target']]

X   = cross_df[feat_cols]
y_v = cross_df['val_target']
y_a = cross_df['aro_target']

X_tr,X_vl,yv_tr,yv_vl = train_test_split(X, y_v, test_size=0.2, random_state=42)
_,  _,  ya_tr,ya_vl   = train_test_split(X, y_a, test_size=0.2, random_state=42)

print("\nTraining XGBoost for 2b...")
xgb_v = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4,
                       random_state=42, verbosity=0)
xgb_a = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4,
                       random_state=42, verbosity=0)
xgb_v.fit(X_tr, yv_tr)
xgb_a.fit(X_tr, ya_tr)

r_v, _ = pearsonr(yv_vl, xgb_v.predict(X_vl))
r_a, _ = pearsonr(ya_vl, xgb_a.predict(X_vl))
print(f"Validation → Val r: {r_v:.3f}  Aro r: {r_a:.3f}")

# Retrain on full data
xgb_v.fit(X, y_v)
xgb_a.fit(X, y_a)

# ── OFFICIAL TEST SCORING ──────────────────────────────
labels2 = pd.read_csv(r'D:\MLproject semeval\data\test_labels_subtask2a_and_2b.csv')
test2   = pd.read_csv(r'D:\MLproject semeval\data\test_subtask2.csv')

pv, pa, tv, ta = [], [], [], []

for _, urow in test2.iterrows():
    uid  = urow['user_id']
    ulbl = labels2[labels2['user_id'] == uid]
    uw   = wave_features[wave_features['user_id'] == uid].sort_values(wave_col)
    if len(uw) == 0 or len(ulbl) == 0:
        continue

    curr = uw.iloc[-1]
    row  = pd.DataFrame([{
        'curr_wave':      curr[wave_col],
        'curr_mean_val':  curr['mean_valence'],
        'curr_mean_aro':  curr['mean_arousal'],
        'curr_std_val':   curr['std_valence'],
        'curr_std_aro':   curr['std_arousal'],
        'curr_min_val':   curr['min_valence'],
        'curr_max_val':   curr['max_valence'],
        'curr_sentiment': curr['mean_sentiment'],
        'curr_pos':       curr['mean_pos_ratio'],
        'curr_neg':       curr['mean_neg_ratio'],
        'curr_wc':        curr['mean_wc'],
        'curr_fp':        curr['mean_fp'],
        'curr_tc':        curr['text_count'],
        'hist_mean_val':  uw['mean_valence'].mean(),
        'hist_mean_aro':  uw['mean_arousal'].mean(),
        'hist_val_trend': uw['mean_valence'].diff().mean(),
        'hist_wave_count':len(uw),
    }])

    pv.append(xgb_v.predict(row[feat_cols].fillna(0))[0])
    pa.append(xgb_a.predict(row[feat_cols].fillna(0))[0])
    tv.append(ulbl['disp_change_valence'].values[0])
    ta.append(ulbl['disp_change_arousal'].values[0])

r_vt, _ = pearsonr(tv, pv)
r_at, _ = pearsonr(ta, pa)

print("\n" + "="*50)
print("🎯 SUBTASK 2B OFFICIAL TEST SCORES")
print("="*50)
print(f"Valence forecast r : {r_vt:.3f}")
print(f"Arousal forecast r : {r_at:.3f}")
print("="*50)
print("✅ Screenshot this and send to Claude!")

pd.DataFrame({
    'user_id':       [test2['user_id'].iloc[i] for i in range(len(pv))],
    'pred_val_change': pv,
    'pred_aro_change': pa
}).to_csv(r'D:\MLproject semeval\results\submission_subtask2b.csv', index=False)
print("Saved submission_subtask2b.csv")