"""
Run this ONCE in your training notebook/environment, using the RAW
`train` and `test` dataframes (i.e. straight after they're merged, BEFORE
your step-1 cleanup -- fit_preprocessing() replicates steps 1-4 itself).

It saves a single `pipeline_artifacts.pkl` containing every fitted object
and lookup table the Streamlit app needs to transform new rows exactly
the way your training pipeline did. Your already-saved
`/kaggle/working/lgb_gbdt.txt` model file is used as-is; nothing about
your LightGBM training changes.

Usage (inside your notebook, after loading the raw merged train/test):

    from fraud_pipeline import fit_preprocessing
    import pickle

    train_processed, test_processed, artifacts = fit_preprocessing(train_raw, test_raw)

    with open('pipeline_artifacts.pkl', 'wb') as f:
        pickle.dump(artifacts, f)

    print('Saved pipeline_artifacts.pkl with', len(artifacts['FEATURES']), 'features')

Sanity check: `artifacts['FEATURES']` should be identical (same columns,
same order doesn't matter since app.py re-indexes by name) to the
`FEATURES` list you used to train `lgb_model`. If your notebook did any
additional ad-hoc column drops between step 4 and model training that
aren't reflected in fraud_pipeline.py, replicate them in
fraud_pipeline.py's fit_preprocessing/transform_new so train and serve
stay in sync.
"""
import argparse
import pickle
import pandas as pd
from fraud_pipeline import fit_preprocessing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True, help='Path to raw merged train CSV')
    parser.add_argument('--test_csv', required=True, help='Path to raw merged test CSV')
    parser.add_argument('--out', default='pipeline_artifacts.pkl')
    args = parser.parse_args()

    train_raw = pd.read_csv(args.train_csv)
    test_raw = pd.read_csv(args.test_csv)

    _, _, artifacts = fit_preprocessing(train_raw, test_raw)

    with open(args.out, 'wb') as f:
        pickle.dump(artifacts, f)

    print(f"Saved {args.out}")
    print(f"Feature count: {len(artifacts['FEATURES'])}")
    print(f"Drop cols: {len(artifacts['drop_cols'])} | Low-var cols: {len(artifacts['low_var_cols'])}")
    print(f"V-PCA groups: {len(artifacts['v_groups'])}")
    print(f"Categorical/label-encoded cols: {len(artifacts['cat_cols'])}")


if __name__ == '__main__':
    main()
