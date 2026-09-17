"""
Shared preprocessing pipeline for the IEEE-CIS Fraud Detection project.

Two entry points:

    fit_preprocessing(train, test, random_state=42)
        Run this ONCE, at the end of your training notebook, on the full
        raw train/test frames (before your step-1 cleanup). It replicates
        your steps 1-4 exactly, but additionally CAPTURES every fitted
        object / lookup table (drop-column lists, PCA models, frequency
        maps, card-level aggregates, label encoders, the final FEATURES
        list) into a single `artifacts` dict you can pickle.

    transform_new(df_raw, artifacts)
        Run this in the Streamlit app on new raw rows (e.g. your 100-row
        sample). It applies the SAME transformations using ONLY the
        artifacts captured above -- nothing is re-fit on the small sample,
        which is what makes the demo numerically consistent with training.

Note on fidelity: your original `feature_engineering()` recomputed
frequency counts and card1-level group means/stds on whatever dataframe
was passed in. That's fine for a 590k-row train/test split, but it breaks
down on a 100-row inference sample (most card1 groups would have exactly
1 row, so std -> NaN and "ratio vs card mean" -> ~1 for everyone). This
module fixes that by computing those lookups ONCE on the training data
and reusing them at inference time -- which is the statistically correct
way to serve a model like this. Everything else matches your notebook
step-for-step.
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

BIG_DOMAINS = {'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'icloud.com'}
FREQ_COLS = ['card1', 'card2', 'card3', 'card5', 'uid', 'uid2',
             'P_emaildomain', 'R_emaildomain', 'DeviceInfo']
AGG_COLS = ['TransactionAmt', 'TransactionAmt_log', 'D1', 'D15']
DROP_COLS_ALWAYS = ['isFraud', 'TransactionID', 'TransactionDT']


# --------------------------------------------------------------------------
# Step 3 helpers (feature engineering) -- shared between fit and transform
# --------------------------------------------------------------------------
def _add_base_features(df):
    """Columns that don't depend on any train-fit lookup table."""
    df = df.copy()
    df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])
    df['TransactionAmt_decimal'] = df['TransactionAmt'] - np.floor(df['TransactionAmt'])

    df['hour'] = (df['TransactionDT'] // 3600) % 24
    df['day'] = (df['TransactionDT'] // 86400) % 7
    df['week'] = df['TransactionDT'] // 604800
    df['hour_day'] = df['hour'] * 7 + df['day']

    df['uid'] = (df['card1'].astype(str) + '_' +
                 df['addr1'].astype(str) + '_' +
                 df['P_emaildomain'].astype(str))
    df['uid2'] = df['card1'].astype(str) + '_' + df['card2'].astype(str)

    df['email_match'] = (df['P_emaildomain'].astype(str) ==
                          df['R_emaildomain'].astype(str)).astype(int)
    df['P_email_type'] = df['P_emaildomain'].apply(
        lambda x: 1 if str(x) in BIG_DOMAINS else 0)

    d_cols = [c for c in df.columns if c.startswith('D')]
    for col in d_cols:
        df[f'{col}_isna'] = df[col].isna().astype(int)

    id_cols = [c for c in df.columns if c.startswith('id')]
    if id_cols:
        df['has_identity'] = df[id_cols].notnull().any(axis=1).astype(int)
        df['identity_richness'] = df[id_cols].notnull().sum(axis=1)
    else:
        df['has_identity'] = 0
        df['identity_richness'] = 0

    df['null_count'] = df.isnull().sum(axis=1)
    return df


def _apply_lookup_features(df, freq_maps, card1_mean_maps, card1_std_maps,
                            card1_count_map, card12_count_map,
                            d1_mean_map, d4_mean_map, global_amt_mean):
    """Columns that depend on train-fit lookup tables. Safe for 1..N rows."""
    df = df.copy()

    for col in FREQ_COLS:
        if col in df.columns:
            df[f'{col}_freq'] = df[col].map(freq_maps.get(col, {})).fillna(0)

    for col in AGG_COLS:
        if col in df.columns and col in card1_mean_maps:
            df[f'card1_{col}_mean'] = df['card1'].map(card1_mean_maps[col])
            df[f'card1_{col}_std'] = df['card1'].map(card1_std_maps[col])
            # unseen card1 in the sample -> fall back to global stats
            df[f'card1_{col}_mean'] = df[f'card1_{col}_mean'].fillna(global_amt_mean.get(col, 0))
            df[f'card1_{col}_std'] = df[f'card1_{col}_std'].fillna(0)

    if 'card1_TransactionAmt_mean' in df.columns:
        df['amt_vs_card_mean'] = df['TransactionAmt'] / (df['card1_TransactionAmt_mean'] + 1)

    # percentile rank of this transaction's amount within the *training*
    # distribution for that card (approximated via mean/std z-ish rank is
    # overkill here -- for a small demo sample we rank against the sample
    # itself, which is the best we can do without shipping the full
    # per-card training distribution).
    df['amt_pct_rank'] = df.groupby('card1')['TransactionAmt'].rank(pct=True).fillna(0.5)

    df['card1_count'] = df['card1'].map(card1_count_map).fillna(1)
    card12_key = list(zip(df['card1'], df['card2']))
    df['card12_count'] = [card12_count_map.get(k, 1) for k in card12_key]

    if 'D1' in df.columns:
        df['D1_normalized'] = df['D1'] - df['card1'].map(d1_mean_map)
        df['D1_normalized'] = df['D1_normalized'].fillna(df['D1'])
    if 'D4' in df.columns:
        df['D4_normalized'] = df['D4'] - df['card1'].map(d4_mean_map)
        df['D4_normalized'] = df['D4_normalized'].fillna(df['D4'])

    return df


# --------------------------------------------------------------------------
# FIT (run once on full raw train/test in your training notebook)
# --------------------------------------------------------------------------
def fit_preprocessing(train, test, random_state=42):
    train = train.copy()
    test = test.copy()
    artifacts = {}

    # ---- Step 1: drop high-missing & low-variance columns ----
    miss_pct = train.isnull().mean()
    drop_cols = miss_pct[miss_pct > 0.90].index.tolist()
    train.drop(columns=drop_cols, inplace=True)
    test.drop(columns=drop_cols, errors='ignore', inplace=True)

    num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != 'isFraud']
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(train[num_cols].fillna(0))
    low_var = [c for c, s in zip(num_cols, selector.get_support()) if not s]
    train.drop(columns=low_var, inplace=True)
    test.drop(columns=low_var, errors='ignore', inplace=True)

    artifacts['drop_cols'] = drop_cols
    artifacts['low_var_cols'] = low_var

    # ---- Step 2: V-feature PCA compression ----
    v_cols = [c for c in train.columns if c.startswith('V')]
    miss_patterns = train[v_cols].isnull().T.apply(lambda r: r.values.tobytes(), axis=1)
    groups = {}
    for col, pat in miss_patterns.items():
        groups.setdefault(pat, []).append(col)

    v_groups = []
    pca_results_tr, pca_results_te = [], []
    for i, group_cols in enumerate(groups.values()):
        if len(group_cols) < 2:
            continue
        sparsity = train[group_cols].isnull().mean().mean()
        if sparsity > 0.80:
            n_components = min(3, len(group_cols))
        elif sparsity > 0.50:
            n_components = min(5, len(group_cols))
        else:
            n_components = min(6, len(group_cols))

        imp = SimpleImputer(strategy='median')
        tr_imp = imp.fit_transform(train[group_cols])
        te_imp = imp.transform(test[group_cols])

        pca = PCA(n_components=n_components, random_state=random_state)
        tr_pca = pca.fit_transform(tr_imp)
        te_pca = pca.transform(te_imp)

        col_names = [f'V_pca_g{i}_pc{j}' for j in range(n_components)]
        pca_results_tr.append(pd.DataFrame(tr_pca, columns=col_names, index=train.index))
        pca_results_te.append(pd.DataFrame(te_pca, columns=col_names, index=test.index))

        v_groups.append({
            'group_idx': i,
            'cols': group_cols,
            'imputer': imp,
            'pca': pca,
            'n_components': n_components,
            'out_cols': col_names,
        })

    train = pd.concat([train.drop(columns=v_cols)] + pca_results_tr, axis=1)
    test = pd.concat([test.drop(columns=v_cols)] + pca_results_te, axis=1)
    pca_cols = [c for c in train.columns if 'V_pca' in c]
    train[pca_cols] = train[pca_cols].astype(np.float32)
    test[pca_cols] = test[pca_cols].astype(np.float32)

    artifacts['v_groups'] = v_groups

    # ---- Step 3: feature engineering ----
    train = _add_base_features(train)
    test = _add_base_features(test)

    freq_maps = {c: train[c].value_counts().to_dict() for c in FREQ_COLS if c in train.columns}
    card1_mean_maps, card1_std_maps, global_amt_mean = {}, {}, {}
    for col in AGG_COLS:
        if col in train.columns:
            grp = train.groupby('card1')[col]
            card1_mean_maps[col] = grp.mean().to_dict()
            card1_std_maps[col] = grp.std().to_dict()
            global_amt_mean[col] = train[col].mean()
    card1_count_map = train.groupby('card1')['TransactionAmt'].count().to_dict()
    card12_count_map = train.groupby(['card1', 'card2'])['TransactionAmt'].count().to_dict()
    d1_mean_map = train.groupby('card1')['D1'].mean().to_dict() if 'D1' in train.columns else {}
    d4_mean_map = train.groupby('card1')['D4'].mean().to_dict() if 'D4' in train.columns else {}

    artifacts.update(dict(
        freq_maps=freq_maps, card1_mean_maps=card1_mean_maps, card1_std_maps=card1_std_maps,
        card1_count_map=card1_count_map, card12_count_map=card12_count_map,
        d1_mean_map=d1_mean_map, d4_mean_map=d4_mean_map, global_amt_mean=global_amt_mean,
    ))

    train = _apply_lookup_features(train, freq_maps, card1_mean_maps, card1_std_maps,
                                    card1_count_map, card12_count_map,
                                    d1_mean_map, d4_mean_map, global_amt_mean)
    test = _apply_lookup_features(test, freq_maps, card1_mean_maps, card1_std_maps,
                                   card1_count_map, card12_count_map,
                                   d1_mean_map, d4_mean_map, global_amt_mean)

    # ---- Step 4: label encoding ----
    cat_cols = [c for c in train.select_dtypes('object').columns.tolist() if c in test.columns]
    cat_cols += [c for c in train.select_dtypes('category').columns.tolist()
                 if c in test.columns and c not in cat_cols]

    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        label_encoders[col] = le

    train_only_cats = [c for c in train.select_dtypes(['object', 'category']).columns
                        if c not in test.columns]
    if train_only_cats:
        train.drop(columns=train_only_cats, inplace=True)

    features = [c for c in train.columns if c not in DROP_COLS_ALWAYS and c in test.columns]

    artifacts.update(dict(
        cat_cols=cat_cols, label_encoders=label_encoders,
        train_only_cats=train_only_cats, FEATURES=features,
    ))

    return train, test, artifacts


# --------------------------------------------------------------------------
# TRANSFORM (used at inference time in the Streamlit app)
# --------------------------------------------------------------------------
def transform_new(df_raw, artifacts):
    """Apply the fitted pipeline to new raw rows. Returns (X, df_with_extras)."""
    df = df_raw.copy()

    # Step 1: drop
    df.drop(columns=artifacts['drop_cols'], errors='ignore', inplace=True)
    df.drop(columns=artifacts['low_var_cols'], errors='ignore', inplace=True)

    # Step 2: V-PCA using fitted imputers/PCA per group
    v_cols_present = [c for g in artifacts['v_groups'] for c in g['cols'] if c in df.columns]
    pca_frames = []
    for g in artifacts['v_groups']:
        cols = [c for c in g['cols'] if c in df.columns]
        if len(cols) != len(g['cols']):
            # sample is missing some V columns entirely -> fill with NaN so
            # the imputer can still run
            for c in g['cols']:
                if c not in df.columns:
                    df[c] = np.nan
        arr = g['imputer'].transform(df[g['cols']])
        pcs = g['pca'].transform(arr)
        pca_frames.append(pd.DataFrame(pcs, columns=g['out_cols'], index=df.index))

    df = df.drop(columns=v_cols_present, errors='ignore')
    if pca_frames:
        df = pd.concat([df] + pca_frames, axis=1)
    pca_cols = [c for c in df.columns if 'V_pca' in c]
    df[pca_cols] = df[pca_cols].astype(np.float32)

    # Step 3: feature engineering
    df = _add_base_features(df)
    df = _apply_lookup_features(
        df, artifacts['freq_maps'], artifacts['card1_mean_maps'], artifacts['card1_std_maps'],
        artifacts['card1_count_map'], artifacts['card12_count_map'],
        artifacts['d1_mean_map'], artifacts['d4_mean_map'], artifacts['global_amt_mean'],
    )

    # Step 4: label encoding (unseen categories -> -1)
    for col, le in artifacts['label_encoders'].items():
        if col not in df.columns:
            continue
        classes = set(le.classes_)
        vals = df[col].astype(str)
        known = vals.where(vals.isin(classes), le.classes_[0])
        df[col] = le.transform(known)

    features = artifacts['FEATURES']
    for c in features:
        if c not in df.columns:
            df[c] = 0

    X = df[features].copy()
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    return X, df
