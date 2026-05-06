"""
Train the CACEIS attrition risk model.

Output: models/attrition_model.pkl (model + encoders + metrics + feature importance)
        data/cohort_risk.pkl, country_risk.pkl, concentration.pkl

Usage: python src/train_model.py
       (looks for raw data in CACEIS_DATA_DIR env var, else ./data/raw/)
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_RAW = Path(os.environ.get('CACEIS_DATA_DIR', ROOT / 'data' / 'raw'))
MODEL_DIR = ROOT / 'models'
DATA_DIR = ROOT / 'data'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"Data dir: {DATA_RAW}")
print(f"Model dir: {MODEL_DIR}")

# ── Load HR master ────────────────────────────────────────────────────────────
hr = pd.read_excel(DATA_RAW / 'Data.xlsx', sheet_name='Sheet1',
                   dtype={'ID Employee': str}, engine='openpyxl')
hr.columns = [c.strip() for c in hr.columns]
hr['PERIOD'] = pd.to_datetime(hr['PERIOD'])
if 'ID Employee.1' in hr.columns:
    hr.drop(columns=['ID Employee.1'], inplace=True)

print(f"HR Master: {len(hr):,} rows · {hr['PERIOD'].dt.year.min()}–{hr['PERIOD'].dt.year.max()}")

# ── Build target: exit within 3 months ────────────────────────────────────────
periods = sorted(hr['PERIOD'].unique())
presence = {p: set(hr[hr['PERIOD'] == p]['ID Employee'].dropna().astype(str)) for p in periods}

records = []
for i, p in enumerate(periods[:-3]):
    snap = hr[hr['PERIOD'] == p].copy()
    future = presence[periods[i+1]] | presence[periods[i+2]] | presence[periods[i+3]]
    snap['exits_3m'] = (~snap['ID Employee'].astype(str).isin(future)).astype(int)
    records.append(snap)
panel = pd.concat(records, ignore_index=True)

# ── Feature engineering ───────────────────────────────────────────────────────
for col in ['DATE_ENTRY_GROUP', 'DATE_ENTRY_CACEIS', 'DATE_ENTRY_POSTE']:
    panel[col] = pd.to_datetime(panel[col], errors='coerce')

panel['tenure_group_months']  = ((panel['PERIOD'] - panel['DATE_ENTRY_GROUP']).dt.days / 30.44).clip(lower=0)
panel['tenure_caceis_months'] = ((panel['PERIOD'] - panel['DATE_ENTRY_CACEIS']).dt.days / 30.44).clip(lower=0)
panel['tenure_poste_months']  = ((panel['PERIOD'] - panel['DATE_ENTRY_POSTE']).dt.days / 30.44).clip(lower=0)
panel['period_year']  = panel['PERIOD'].dt.year
panel['period_month'] = panel['PERIOD'].dt.month

features_cat = ['COUNTRY_GROUP_LABEL_EN', 'Age range', 'SEXE_GROUP_LABEL_EN',
                'CONTRACT_GROUP_LABEL_EN', 'DEGREE_LEVEL_GROUP_LABEL_EN',
                'REASON_ENTRY_GROUP_LABEL_EN', 'ENTITY_LABEL_LOCAL']
features_num = ['tenure_group_months', 'tenure_caceis_months', 'tenure_poste_months',
                'period_year', 'period_month']

panel_clean = panel[features_cat + features_num + ['exits_3m', 'PERIOD']].copy()
for c in features_cat:
    panel_clean[c] = panel_clean[c].astype(str).fillna('UNK')
for c in features_num:
    panel_clean[c] = panel_clean[c].fillna(panel_clean[c].median())

print(f"Panel: {len(panel_clean):,} rows · {panel_clean['exits_3m'].mean()*100:.2f}% exit rate")

# ── Temporal train/test split ─────────────────────────────────────────────────
cutoff = pd.Timestamp('2025-04-01')
train = panel_clean[panel_clean['PERIOD'] < cutoff].copy()
test  = panel_clean[panel_clean['PERIOD'] >= cutoff].copy()

# ── Encode ────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder
encoders = {}
for c in features_cat:
    le = LabelEncoder()
    le.fit(panel_clean[c].astype(str))
    train[c] = le.transform(train[c].astype(str))
    test[c]  = le.transform(test[c].astype(str))
    encoders[c] = le

X_train = train[features_cat + features_num]
y_train = train['exits_3m']
X_test  = test[features_cat + features_num]
y_test  = test['exits_3m']

# ── Train ─────────────────────────────────────────────────────────────────────
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

scale = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Class imbalance ratio: {scale:.1f}")

model = xgb.XGBClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    scale_pos_weight=scale, subsample=0.85, colsample_bytree=0.85,
    eval_metric='aucpr', random_state=42, n_jobs=-1, tree_method='hist',
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# ── Validate ──────────────────────────────────────────────────────────────────
proba_test = model.predict_proba(X_test)[:, 1]
proba_train = model.predict_proba(X_train)[:, 1]
metrics = {
    'auc_train': float(roc_auc_score(y_train, proba_train)),
    'auc_test':  float(roc_auc_score(y_test, proba_test)),
    'ap_train':  float(average_precision_score(y_train, proba_train)),
    'ap_test':   float(average_precision_score(y_test, proba_test)),
    'n_train':   int(len(train)),
    'n_test':    int(len(test)),
    'exit_rate_train': float(y_train.mean()),
    'exit_rate_test':  float(y_test.mean()),
}
print(f"\nAUC test: {metrics['auc_test']:.3f}")
print(f"AP test: {metrics['ap_test']:.3f}")
print(classification_report(y_test, (proba_test > 0.5).astype(int), digits=3))

fi = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_}) \
    .sort_values('importance', ascending=False)

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump({
    'model': model, 'encoders': encoders,
    'features_cat': features_cat, 'features_num': features_num,
    'metrics': metrics, 'feature_importance': fi.to_dict('records'),
}, MODEL_DIR / 'attrition_model.pkl')
print(f"Saved: {MODEL_DIR / 'attrition_model.pkl'}")

# ── Generate cohort risk for cockpit ──────────────────────────────────────────
latest = hr['PERIOD'].max()
snap = hr[hr['PERIOD'] == latest].copy()
for col in ['DATE_ENTRY_GROUP', 'DATE_ENTRY_CACEIS', 'DATE_ENTRY_POSTE']:
    snap[col] = pd.to_datetime(snap[col], errors='coerce')
snap['tenure_group_months']  = ((snap['PERIOD'] - snap['DATE_ENTRY_GROUP']).dt.days / 30.44).clip(lower=0)
snap['tenure_caceis_months'] = ((snap['PERIOD'] - snap['DATE_ENTRY_CACEIS']).dt.days / 30.44).clip(lower=0)
snap['tenure_poste_months']  = ((snap['PERIOD'] - snap['DATE_ENTRY_POSTE']).dt.days / 30.44).clip(lower=0)
snap['period_year']  = snap['PERIOD'].dt.year
snap['period_month'] = snap['PERIOD'].dt.month

X = snap.copy()
for c in features_cat:
    X[c] = X[c].astype(str).fillna('UNK')
    known = set(encoders[c].classes_)
    X[c] = X[c].apply(lambda v: v if v in known else encoders[c].classes_[0])
    X[c] = encoders[c].transform(X[c].astype(str))
for c in features_num:
    X[c] = X[c].fillna(X[c].median())
snap['attrition_risk'] = model.predict_proba(X[features_cat + features_num])[:, 1]

cohorts = (snap.groupby(['COUNTRY_GROUP_LABEL_EN', 'CONTRACT_GROUP_LABEL_EN', 'Age range'], dropna=False)
    .agg(n=('ID Employee', 'count'),
         risk_mean=('attrition_risk', 'mean'),
         risk_max=('attrition_risk', 'max'),
         tenure_avg=('tenure_caceis_months', 'mean'))
    .reset_index().sort_values('risk_mean', ascending=False))
cohorts = cohorts[cohorts['n'] >= 5].copy()
cohorts['risk_band'] = pd.cut(cohorts['risk_mean'], bins=[0, 0.05, 0.15, 0.30, 1.0],
                              labels=['Low', 'Medium', 'High', 'Critical'])
cohorts.to_pickle(DATA_DIR / 'cohort_risk.pkl')

country = snap.groupby('COUNTRY_GROUP_LABEL_EN').agg(
    n=('ID Employee', 'count'),
    risk_mean=('attrition_risk', 'mean'),
    risk_p90=('attrition_risk', lambda x: x.quantile(0.90)),
).reset_index().sort_values('risk_mean', ascending=False)
country.to_pickle(DATA_DIR / 'country_risk.pkl')

cohorts['concentration_score'] = cohorts['risk_mean'] * (cohorts['tenure_avg'] / 12).clip(upper=10)
top_conc = cohorts.sort_values('concentration_score', ascending=False).head(15)
top_conc.to_pickle(DATA_DIR / 'concentration.pkl')

print(f"\nDone. {len(cohorts)} cohorts, {len(country)} countries.")
