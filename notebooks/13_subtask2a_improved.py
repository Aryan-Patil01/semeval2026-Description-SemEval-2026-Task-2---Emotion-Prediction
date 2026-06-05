"""
SemEval Subtask 2A — Final Solution
=====================================
KEY INSIGHT:
  test_subtask2.csv has NO text column — only (user_id, timestamp_min, timestamp_max).
  The actual diary entries live in train_subtask1.csv + test_subtask1.csv.
  For each subtask2 row we find ALL diary entries for that user within
  [timestamp_min, timestamp_max] and aggregate features from them.
  That aggregated feature vector is what we train/predict on.

PIPELINE:
  1. Load subtask1 diary pool (train + test combined)
  2. Load subtask2 train + test + labels
  3. For every subtask2 row → aggregate features from matching diary window
  4. Train XGBoost + LightGBM ensemble with 5-fold CV
  5. Predict on test, score vs labels, save submission
"""

import pandas as pd
import numpy as np
import warnings, re, os
from nltk.sentiment import SentimentIntensityAnalyzer
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from scipy.stats import pearsonr
import nltk

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("⚠  LightGBM not found — XGBoost only. pip install lightgbm")

nltk.download('vader_lexicon', quiet=True)
warnings.filterwarnings('ignore')

# ── PATHS ── edit these if your layout differs ──────────────────────────────
DATA_DIR    = r'D:\MLproject semeval\data'
RESULTS_DIR = r'D:\MLproject semeval\results'
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_S1   = os.path.join(DATA_DIR, 'train_subtask1.csv')
TEST_S1    = os.path.join(DATA_DIR, 'test_subtask1.csv')
TRAIN_S2   = os.path.join(DATA_DIR, 'train_subtask2a.csv')
TEST_S2    = os.path.join(DATA_DIR, 'test_subtask2.csv')
TEST_LBLS  = os.path.join(DATA_DIR, 'test_labels_subtask2a_and_2b.csv')
OUTPUT_CSV = os.path.join(RESULTS_DIR, 'submission_subtask2a_final.csv')
# ────────────────────────────────────────────────────────────────────────────


# ============================================================
# 1.  LOAD DIARY POOL  (subtask1 train + test combined)
# ============================================================
print("Loading diary pool (subtask1)...")
s1_frames = []
for path, split in [(TRAIN_S1, 'train'), (TEST_S1, 'test')]:
    try:
        tmp = pd.read_csv(path)
        tmp['_split'] = split
        s1_frames.append(tmp)
        print(f"  {split}: {len(tmp)} rows,  cols: {list(tmp.columns)}")
    except FileNotFoundError:
        print(f"  ⚠  {path} not found — skipping")

diary = pd.concat(s1_frames, ignore_index=True)

# Normalise column names (lower-case, strip spaces)
diary.columns = [c.strip().lower().replace(' ', '_') for c in diary.columns]
print(f"Diary pool total: {len(diary)} rows")
print(f"Diary columns: {list(diary.columns)}")

# Find text column
TEXT_COL = None
for candidate in ['text', 'post', 'content', 'entry', 'message', 'body']:
    if candidate in diary.columns:
        TEXT_COL = candidate
        break
if TEXT_COL is None:
    # pick the longest string column heuristically
    str_cols = diary.select_dtypes(include='object').columns.tolist()
    TEXT_COL = max(str_cols, key=lambda c: diary[c].dropna().str.len().mean())
    print(f"  ⚠  Guessing text column: '{TEXT_COL}'")
else:
    print(f"  Text column: '{TEXT_COL}'")

# Timestamp column in diary
TS_COL_D = None
for candidate in ['timestamp', 'date', 'created_at', 'time']:
    if candidate in diary.columns:
        TS_COL_D = candidate
        break
if TS_COL_D is None:
    raise ValueError("Cannot find timestamp column in diary data. "
                     f"Available: {list(diary.columns)}")
diary['_ts'] = pd.to_datetime(diary[TS_COL_D], errors='coerce')

# User id column in diary
UID_COL_D = None
for candidate in ['user_id', 'userid', 'user', 'author_id', 'author']:
    if candidate in diary.columns:
        UID_COL_D = candidate
        break
if UID_COL_D is None:
    raise ValueError(f"Cannot find user_id column in diary. Available: {list(diary.columns)}")

diary = diary.dropna(subset=['_ts']).sort_values(['_ts']).reset_index(drop=True)

# Valence / arousal columns in diary (for state features)
VAL_COL_D = next((c for c in diary.columns if 'valence' in c), None)
ARO_COL_D = next((c for c in diary.columns if 'arousal' in c), None)
if VAL_COL_D:
    diary[VAL_COL_D] = pd.to_numeric(diary[VAL_COL_D], errors='coerce')
    diary[VAL_COL_D] = diary[VAL_COL_D] + 2   # shift to non-negative
if ARO_COL_D:
    diary[ARO_COL_D] = pd.to_numeric(diary[ARO_COL_D], errors='coerce')
    diary[ARO_COL_D] = diary[ARO_COL_D] + 1


# ============================================================
# 2.  LOAD SUBTASK 2 DATA
# ============================================================
print("\nLoading subtask2 data...")
tr2   = pd.read_csv(TRAIN_S2)
test2 = pd.read_csv(TEST_S2)
lbls  = pd.read_csv(TEST_LBLS)

tr2.columns   = [c.strip().lower() for c in tr2.columns]
test2.columns = [c.strip().lower() for c in test2.columns]
lbls.columns  = [c.strip().lower() for c in lbls.columns]

print(f"Train subtask2 cols: {list(tr2.columns)}")
print(f"Test  subtask2 cols: {list(test2.columns)}")
print(f"Labels        cols: {list(lbls.columns)}")

# Identify min/max timestamp columns
def find_ts_cols(df):
    """Return (ts_min_col, ts_max_col) or (ts_col, ts_col) if only one."""
    ts_min = next((c for c in df.columns if 'min' in c and 'time' in c), None)
    ts_max = next((c for c in df.columns if 'max' in c and 'time' in c), None)
    if ts_min and ts_max:
        return ts_min, ts_max
    single = next((c for c in df.columns if 'time' in c or 'date' in c), None)
    return single, single

TR_TS_MIN, TR_TS_MAX     = find_ts_cols(tr2)
TEST_TS_MIN, TEST_TS_MAX = find_ts_cols(test2)
print(f"Train window cols : {TR_TS_MIN} → {TR_TS_MAX}")
print(f"Test  window cols : {TEST_TS_MIN} → {TEST_TS_MAX}")

for df_, c1, c2 in [(tr2, TR_TS_MIN, TR_TS_MAX),
                    (test2, TEST_TS_MIN, TEST_TS_MAX)]:
    df_[c1] = pd.to_datetime(df_[c1], errors='coerce')
    if c2 != c1:
        df_[c2] = pd.to_datetime(df_[c2], errors='coerce')

# Target columns in train
TARGET_V = next((c for c in tr2.columns if 'change_valence' in c), None)
TARGET_A = next((c for c in tr2.columns if 'change_arousal' in c), None)
print(f"Train targets     : {TARGET_V}, {TARGET_A}")

# Target columns in test labels
LBL_V = next((c for c in lbls.columns if 'change_valence' in c), None)
LBL_A = next((c for c in lbls.columns if 'change_arousal' in c), None)
print(f"Label targets     : {LBL_V}, {LBL_A}")


# ============================================================
# 3.  TEXT FEATURE EXTRACTOR
# ============================================================
sia = SentimentIntensityAnalyzer()
NEGATION = {'not', "n't", 'never', 'no', 'nobody', 'nothing',
            'nowhere', 'neither', 'nor', 'cannot', 'cant'}
EMOTION_POSITIVE = {'happy', 'joy', 'love', 'great', 'wonderful', 'amazing',
                    'fantastic', 'glad', 'excited', 'grateful', 'blessed',
                    'good', 'awesome', 'nice', 'pleased', 'delighted'}
EMOTION_NEGATIVE = {'sad', 'angry', 'hate', 'terrible', 'horrible', 'awful',
                    'bad', 'worst', 'depressed', 'miserable', 'frustrated',
                    'anxious', 'worried', 'scared', 'upset', 'hurt', 'pain'}

def extract_text_features(text: str) -> dict:
    text   = str(text) if text and str(text) != 'nan' else ''
    tokens = text.lower().split()
    vader  = sia.polarity_scores(text) if text else {'compound':0,'pos':0,'neg':0,'neu':1}
    caps   = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    pos_em = sum(1 for t in tokens if t in EMOTION_POSITIVE)
    neg_em = sum(1 for t in tokens if t in EMOTION_NEGATIVE)
    return {
        'word_count':          len(tokens),
        'avg_word_length':     float(np.mean([len(w) for w in tokens])) if tokens else 0.0,
        'unique_word_ratio':   len(set(tokens)) / max(len(tokens), 1),
        'vader_compound':      vader['compound'],
        'vader_pos':           vader['pos'],
        'vader_neg':           vader['neg'],
        'vader_neu':           vader['neu'],
        'first_person_count':  sum(tokens.count(p) for p in ['i','me','my','myself','i\'m']),
        'exclamation_count':   text.count('!'),
        'question_count':      text.count('?'),
        'ellipsis_count':      text.count('...') + text.count('…'),
        'negation_count':      sum(1 for t in tokens if t in NEGATION),
        'caps_ratio':          caps,
        'positive_emotion_wds': pos_em,
        'negative_emotion_wds': neg_em,
        'emotion_word_ratio':  (pos_em + neg_em) / max(len(tokens), 1),
        'text_length':         len(text),
    }

TEXT_FEAT_NAMES = list(extract_text_features('test').keys())


# ============================================================
# 4.  AGGREGATE DIARY ENTRIES FOR A GIVEN (user, ts_min, ts_max)
# ============================================================
# Pre-group diary by user for speed
diary_by_user = {uid: grp for uid, grp in diary.groupby(UID_COL_D)}

def aggregate_window_features(user_id, ts_min, ts_max,
                               collection_phase=0) -> dict:
    """
    Return a flat feature dict for one subtask2 row.
    Looks up all diary entries for user_id in [ts_min, ts_max].
    Falls back to all user entries if window yields nothing.
    """
    feats = {}
    user_diary = diary_by_user.get(user_id, pd.DataFrame())

    if len(user_diary) == 0:
        # No diary data at all — zero-fill
        for fn in TEXT_FEAT_NAMES:
            for sfx in ('_mean', '_std', '_first', '_last', '_trend'):
                feats[f'{fn}{sfx}'] = 0.0
        feats.update({
            'n_entries': 0, 'window_days': 0,
            'collection_phase': collection_phase,
            'val_mean': 2.0, 'val_std': 0.0, 'val_first': 2.0, 'val_last': 2.0,
            'val_change': 0.0, 'aro_mean': 1.0, 'aro_std': 0.0,
            'aro_first': 1.0, 'aro_last': 1.0, 'aro_change': 0.0,
            'val_aro_corr': 0.0,
        })
        return feats

    # Filter to window
    win = user_diary
    if ts_min is not None and not pd.isna(ts_min):
        win = win[win['_ts'] >= ts_min]
    if ts_max is not None and not pd.isna(ts_max):
        win = win[win['_ts'] <= ts_max]

    # Fall back to all entries if window empty
    if len(win) == 0:
        win = user_diary

    win = win.sort_values('_ts').reset_index(drop=True)
    n   = len(win)

    # — Text features aggregated across all entries in window —
    all_tf = win[TEXT_COL].apply(extract_text_features).apply(pd.Series)
    for fn in TEXT_FEAT_NAMES:
        col_data = all_tf[fn] if fn in all_tf.columns else pd.Series([0.0]*n)
        feats[f'{fn}_mean']  = float(col_data.mean())
        feats[f'{fn}_std']   = float(col_data.std()) if n > 1 else 0.0
        feats[f'{fn}_first'] = float(col_data.iloc[0])
        feats[f'{fn}_last']  = float(col_data.iloc[-1])
        # trend: second-half mean minus first-half mean
        if n >= 4:
            mid = n // 2
            feats[f'{fn}_trend'] = float(col_data.iloc[mid:].mean()
                                         - col_data.iloc[:mid].mean())
        else:
            feats[f'{fn}_trend'] = 0.0

    # — Valence / arousal state features —
    if VAL_COL_D and VAL_COL_D in win.columns:
        v = win[VAL_COL_D].dropna()
        feats['val_mean']   = float(v.mean())  if len(v) else 2.0
        feats['val_std']    = float(v.std())   if len(v) > 1 else 0.0
        feats['val_first']  = float(v.iloc[0]) if len(v) else 2.0
        feats['val_last']   = float(v.iloc[-1])if len(v) else 2.0
        feats['val_change'] = feats['val_last'] - feats['val_first']
    else:
        feats.update({'val_mean':2.0,'val_std':0.0,'val_first':2.0,
                      'val_last':2.0,'val_change':0.0})

    if ARO_COL_D and ARO_COL_D in win.columns:
        a = win[ARO_COL_D].dropna()
        feats['aro_mean']   = float(a.mean())  if len(a) else 1.0
        feats['aro_std']    = float(a.std())   if len(a) > 1 else 0.0
        feats['aro_first']  = float(a.iloc[0]) if len(a) else 1.0
        feats['aro_last']   = float(a.iloc[-1])if len(a) else 1.0
        feats['aro_change'] = feats['aro_last'] - feats['aro_first']
    else:
        feats.update({'aro_mean':1.0,'aro_std':0.0,'aro_first':1.0,
                      'aro_last':1.0,'aro_change':0.0})

    if (VAL_COL_D and ARO_COL_D
            and VAL_COL_D in win.columns and ARO_COL_D in win.columns
            and n >= 3):
        try:
            feats['val_aro_corr'] = float(win[VAL_COL_D].corr(win[ARO_COL_D]))
        except Exception:
            feats['val_aro_corr'] = 0.0
    else:
        feats['val_aro_corr'] = 0.0

    # — Window-level features —
    feats['n_entries']        = n
    feats['collection_phase'] = collection_phase
    if (ts_min is not None and ts_max is not None
            and not pd.isna(ts_min) and not pd.isna(ts_max)):
        feats['window_days'] = (ts_max - ts_min).days
    else:
        feats['window_days'] = 0

    return feats


# ============================================================
# 5.  BUILD FEATURE MATRICES FOR TRAIN & TEST
# ============================================================
print("\nBuilding training features (this may take ~1-2 min)...")

def build_features(df_, ts_min_col, ts_max_col, label='data'):
    rows = []
    n = len(df_)
    for i, row in df_.iterrows():
        if i % 200 == 0:
            print(f"  {label}: {i}/{n}", end='\r')
        cp = row.get('collection_phase_min',
             row.get('collection_phase', 0))
        rows.append(aggregate_window_features(
            user_id        = row['user_id'],
            ts_min         = row[ts_min_col],
            ts_max         = row[ts_max_col],
            collection_phase = cp
        ))
    print(f"  {label}: {n}/{n} ✓")
    return pd.DataFrame(rows)

X_train_df = build_features(tr2, TR_TS_MIN, TR_TS_MAX, 'train')
X_test_df  = build_features(test2, TEST_TS_MIN, TEST_TS_MAX, 'test')

FEATURE_COLS = X_train_df.columns.tolist()
print(f"\nFeature count: {len(FEATURE_COLS)}")

X_train = X_train_df[FEATURE_COLS].fillna(0).values
X_test  = X_test_df[FEATURE_COLS].fillna(0).values

# Scale features (RobustScaler handles outliers well)
scaler  = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

y_v = tr2[TARGET_V].fillna(0).values
y_a = tr2[TARGET_A].fillna(0).values

# Filter out rows where we truly have no useful signal (n_entries == 0)
valid_mask = X_train_df['n_entries'].values > 0
print(f"Rows with at least one diary entry: {valid_mask.sum()} / {len(valid_mask)}")

X_tr_fit = X_train[valid_mask]
y_v_fit  = y_v[valid_mask]
y_a_fit  = y_a[valid_mask]


# ============================================================
# 6.  MODEL DEFINITIONS
# ============================================================
def make_xgb(target='v'):
    # Slightly different hyperparams for valence vs arousal
    depth = 4 if target == 'a' else 5
    return XGBRegressor(
        n_estimators=800, learning_rate=0.015, max_depth=depth,
        subsample=0.75, colsample_bytree=0.7, min_child_weight=8,
        reg_alpha=0.2, reg_lambda=2.0, gamma=0.1,
        random_state=42, verbosity=0, n_jobs=-1
    )

def make_lgb(target='v'):
    depth = 4 if target == 'a' else 5
    return LGBMRegressor(
        n_estimators=800, learning_rate=0.015, max_depth=depth,
        num_leaves=25, subsample=0.75, colsample_bytree=0.7,
        min_child_samples=15, reg_alpha=0.2, reg_lambda=2.0,
        random_state=42, verbose=-1, n_jobs=-1
    )


# ============================================================
# 7.  5-FOLD CROSS-VALIDATION
# ============================================================
print("\nRunning 5-fold cross-validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rv, cv_ra = [], []

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_tr_fit)):
    Xtr, Xvl   = X_tr_fit[tr_idx], X_tr_fit[vl_idx]
    yvtr, yvvl = y_v_fit[tr_idx],  y_v_fit[vl_idx]
    yatr, yavl = y_a_fit[tr_idx],  y_a_fit[vl_idx]

    # XGBoost
    xv = make_xgb('v'); xv.fit(Xtr, yvtr)
    xa = make_xgb('a'); xa.fit(Xtr, yatr)
    pv = xv.predict(Xvl)
    pa = xa.predict(Xvl)

    if HAS_LGB:
        lv = make_lgb('v'); lv.fit(Xtr, yvtr)
        la = make_lgb('a'); la.fit(Xtr, yatr)
        pv = 0.5 * pv + 0.5 * lv.predict(Xvl)
        pa = 0.5 * pa + 0.5 * la.predict(Xvl)

    # Clip predictions to realistic range
    pv = np.clip(pv, -4, 4)
    pa = np.clip(pa, -2, 2)

    rv, _ = pearsonr(yvvl, pv)
    ra, _ = pearsonr(yavl, pa)
    cv_rv.append(rv); cv_ra.append(ra)
    print(f"  Fold {fold+1}: Val r={rv:.3f}   Aro r={ra:.3f}")

print(f"\nCV mean → Val r: {np.mean(cv_rv):.3f} ± {np.std(cv_rv):.3f}")
print(f"CV mean → Aro r: {np.mean(cv_ra):.3f} ± {np.std(cv_ra):.3f}")


# ============================================================
# 8.  RETRAIN ON FULL TRAINING DATA
# ============================================================
print("\nRetraining on full data...")
xgb_v = make_xgb('v'); xgb_v.fit(X_tr_fit, y_v_fit)
xgb_a = make_xgb('a'); xgb_a.fit(X_tr_fit, y_a_fit)
if HAS_LGB:
    lgb_v = make_lgb('v'); lgb_v.fit(X_tr_fit, y_v_fit)
    lgb_a = make_lgb('a'); lgb_a.fit(X_tr_fit, y_a_fit)

def predict(X):
    pv = xgb_v.predict(X)
    pa = xgb_a.predict(X)
    if HAS_LGB:
        pv = 0.5 * pv + 0.5 * lgb_v.predict(X)
        pa = 0.5 * pa + 0.5 * lgb_a.predict(X)
    return np.clip(pv, -4, 4), np.clip(pa, -2, 2)


# ============================================================
# 9.  EVALUATE ON TEST SET
# ============================================================
print("\nScoring on official test set...")

# Merge test predictions with labels on user_id
pred_v_all, pred_a_all = predict(X_test)
test2['pred_change_val'] = pred_v_all
test2['pred_change_aro'] = pred_a_all

merged = test2.merge(lbls[['user_id', LBL_V, LBL_A]], on='user_id', how='inner')
print(f"Test rows matched to labels: {len(merged)}")

true_v = merged[LBL_V].values
true_a = merged[LBL_A].values
pv_m   = merged['pred_change_val'].values
pa_m   = merged['pred_change_aro'].values

r_v, _ = pearsonr(true_v, pv_m)
r_a, _ = pearsonr(true_a, pa_m)

print("\n" + "=" * 55)
print("🎯  SUBTASK 2A  —  OFFICIAL TEST SCORES")
print("=" * 55)
print(f"Valence change  r : {r_v:.3f}   (baseline 0.350)")
print(f"Arousal  change r : {r_a:.3f}   (baseline -0.161)")
print(f"Average         r : {(r_v + r_a) / 2:.3f}")
print("=" * 55)


# ============================================================
# 10.  FEATURE IMPORTANCE
# ============================================================
feat_imp = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance_val': xgb_v.feature_importances_,
    'importance_aro': xgb_a.feature_importances_,
}).sort_values('importance_val', ascending=False)

print("\nTop 15 features for VALENCE prediction:")
print(feat_imp[['feature','importance_val']].head(15).to_string(index=False))

print("\nTop 15 features for AROUSAL prediction:")
print(feat_imp[['feature','importance_aro']]
      .sort_values('importance_aro', ascending=False)
      .head(15).to_string(index=False))


# ============================================================
# 11.  SAVE SUBMISSION
# ============================================================
result_df = merged[['user_id', 'pred_change_val', 'pred_change_aro',
                     LBL_V, LBL_A]].copy()
result_df.columns = ['user_id', 'pred_change_valence', 'pred_change_arousal',
                     'true_change_valence', 'true_change_arousal']
result_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved → {OUTPUT_CSV}")
print("Done ✓")