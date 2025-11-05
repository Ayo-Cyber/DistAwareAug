#!/usr/bin/env python
"""
Run DistAwareAug augmentation on a CSV file (or synthetic sample) and compare
classifier performance before and after augmentation.

Usage examples:

# Use synthetic sample (default)
python run_augment_on_csv.py

# Run on your CSV, explicitly listing categorical cols (comma-separated)
python run_augment_on_csv.py --csv data.csv --label target --categorical-cols country,device

# Let the script infer numeric/categorical columns automatically
python run_augment_on_csv.py --csv data.csv --label target

"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from distawareaug import DistAwareAugmentor


def infer_columns(df: pd.DataFrame, label: str):
    # Numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Treat the label as non-feature
    if label in num_cols:
        num_cols.remove(label)

    # Categorical = non-numeric minus label
    cat_cols = [c for c in df.columns.tolist() if c not in num_cols and c != label]

    return num_cols, cat_cols


def prepare_group(df_group: pd.DataFrame, num_cols, label_col, augmentor_params):
    """Augment a single group (categorical combination). Returns augmented DataFrame."""
    X = df_group[num_cols].values
    y = df_group[label_col].values

    # If group has <=1 sample, skip augmentation (can't fit distributions)
    if len(X) < 2:
        return df_group.copy()

    # Impute numeric missing values
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Create augmentor
    augmentor = DistAwareAugmentor(**augmentor_params)

    try:
        X_res, y_res = augmentor.fit_resample(X_scaled, y)
    except Exception as e:
        warnings.warn(f"Augmentation failed for group (skipping): {e}")
        return df_group.copy()

    # Inverse-transform numeric features back to original scale
    # Note: scaler was fit on group, so shape matches
    # X_res may have more rows than original
    X_res_orig = scaler.inverse_transform(X_res)

    # Build DataFrame for resampled rows
    df_res = pd.DataFrame(X_res_orig, columns=num_cols)
    df_res[label_col] = y_res

    return df_res


def run(args):
    # If CSV path provided, load it. Otherwise generate synthetic dataset
    if args.csv:
        p = Path(args.csv)
        if not p.exists():
            print(f"CSV file not found: {args.csv}")
            sys.exit(1)
        df = pd.read_csv(p)
        if args.label not in df.columns:
            print(f"Label column '{args.label}' not found in CSV columns: {df.columns.tolist()}")
            sys.exit(1)
    else:
        # Generate synthetic data for demo
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=7,
            n_redundant=3,
            weights=[0.9, 0.1],
            random_state=args.random_state,
        )
        num_cols = [f"f{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=num_cols)
        df[args.label] = y

    # Determine numeric and categorical columns
    if args.numeric_cols:
        numeric_cols = [c.strip() for c in args.numeric_cols.split(",") if c.strip()]
    else:
        # infer
        numeric_cols, inferred_cat = infer_columns(df, args.label)

    if args.categorical_cols:
        cat_cols = [c.strip() for c in args.categorical_cols.split(",") if c.strip()]
    else:
        # If numeric_cols was passed explicitly, infer cat_cols from remainder
        if args.numeric_cols:
            cat_cols = [c for c in df.columns.tolist() if c not in numeric_cols and c != args.label]
        else:
            cat_cols = inferred_cat

    print("Using numeric columns:", numeric_cols)
    print("Using categorical columns:", cat_cols)

    # Split into train/test
    X = df[numeric_cols]
    y = df[args.label]

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Baseline classifier on numeric features only (scaled)
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    clf = RandomForestClassifier(random_state=args.random_state, n_estimators=50)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    print("\nBaseline classifier results (trained on original imbalanced data):\n")
    print(classification_report(y_test, y_pred))

    # Augmentation parameters
    augmentor_params = {
        "sampling_strategy": args.sampling_strategy,
        "sampling_mode": args.sampling_mode,
        "distribution_method": args.distribution_method,
        "diversity_threshold": args.diversity_threshold,
        "distance_metric": args.distance_metric,
        "random_state": args.random_state,
    }

    # If there are categorical columns, perform group-wise augmentation
    augmented_parts = []

    if cat_cols:
        print("\nPerforming group-wise augmentation by categorical combinations...")
        # Group using provided categorical columns
        grouped = df_train.groupby(cat_cols)
        for group_vals, group_df in grouped:
            # Skip tiny groups
            if len(group_df) < args.min_group_size:
                augmented_parts.append(group_df)
                continue

            df_aug_group = prepare_group(group_df, numeric_cols, args.label, augmentor_params)

            # Add categorical columns back for resampled rows: if group_vals is a scalar, make tuple
            if isinstance(group_vals, tuple):
                for col, val in zip(cat_cols, group_vals):
                    df_aug_group[col] = val
            else:
                df_aug_group[cat_cols[0]] = group_vals

            augmented_parts.append(df_aug_group)

        df_augmented = pd.concat(augmented_parts, ignore_index=True)
    else:
        print("\nNo categorical columns provided — augmenting on whole dataset (numeric features only)...")
        # Build DataFrame with numeric cols + label for train portion
        train_df = pd.concat([df_train[numeric_cols], df_train[[args.label]]], axis=1)
        df_augmented = prepare_group(train_df, numeric_cols, args.label, augmentor_params)

    # If the augmented result has same numeric columns + label, ensure categorical columns exist
    for c in cat_cols:
        if c not in df_augmented.columns:
            df_augmented[c] = df_augmented.get(c, np.nan)

    # Save augmented CSV if requested
    if args.output:
        out_path = Path(args.output)
        df_augmented.to_csv(out_path, index=False)
        print(f"Augmented dataset saved to {out_path}")

    # Train classifier on augmented data
    # Use numeric columns only for classifier training
    X_aug = df_augmented[numeric_cols]
    y_aug = df_augmented[args.label]

    # Impute and scale using parameters fit on augmented data
    imputer_aug = SimpleImputer(strategy="median")
    X_aug_imp = imputer_aug.fit_transform(X_aug)
    X_test_imp2 = imputer_aug.transform(X_test)

    scaler_aug = StandardScaler()
    X_aug_scaled = scaler_aug.fit_transform(X_aug_imp)
    X_test_scaled2 = scaler_aug.transform(X_test_imp2)

    clf2 = RandomForestClassifier(random_state=args.random_state, n_estimators=50)
    clf2.fit(X_aug_scaled, y_aug)
    y_pred2 = clf2.predict(X_test_scaled2)

    print("\nClassifier results after augmentation (trained on augmented data):\n")
    print(classification_report(y_test, y_pred2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DistAwareAug on CSV or sample data")
    parser.add_argument("--csv", type=str, help="Path to CSV file (optional). If omitted, a synthetic dataset is used.")
    parser.add_argument("--label", type=str, default="target", help="Name of the label column")
    parser.add_argument(
        "--numeric-cols",
        type=str,
        default=None,
        help="Comma-separated list of numeric columns. If omitted, they will be inferred.",
    )
    parser.add_argument(
        "--categorical-cols",
        type=str,
        default=None,
        help="Comma-separated list of categorical columns. If omitted, they will be inferred.",
    )
    parser.add_argument("--output", type=str, default=None, help="Path to save augmented CSV")

    parser.add_argument("--distribution-method", type=str, default="gaussian", choices=["kde", "gaussian", "uniform"], help="Distribution fitting method")
    parser.add_argument("--diversity-threshold", type=float, default=0.1, help="Diversity threshold")
    parser.add_argument("--distance-metric", type=str, default="euclidean", help="Distance metric for diversity checks")
    parser.add_argument("--sampling-strategy", type=str, default="auto", help="Sampling strategy (auto or dict-like)")
    parser.add_argument("--sampling-mode", type=str, default="add", choices=["add", "target"], help="How to interpret sampling_strategy")
    parser.add_argument("--min-group-size", type=int, default=5, help="Minimum group size to run augmentation")
    parser.add_argument("--test-size", type=float, default=0.3, help="Test set fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    run(args)
