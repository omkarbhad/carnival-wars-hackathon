import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import platform

# GPU detection
import subprocess

def has_nvidia_gpu():
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except FileNotFoundError:
        return False

USE_GPU = has_nvidia_gpu()
DEVICE = "GPU" if USE_GPU else "CPU"
print(f"🖥️ Training on: {DEVICE}")

# Model imports
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

def clean_and_engineer_features(df):
    df = df.copy()
    df.drop(columns=['Product_id', 'Customer_name'], inplace=True, errors='ignore')

    if 'instock_date' in df.columns:
        df['instock_date'] = pd.to_datetime(df['instock_date'], errors='coerce')
        df['day'] = df['instock_date'].dt.day
        df['month'] = df['instock_date'].dt.month
        df['year'] = df['instock_date'].dt.year
        df['dayofweek'] = df['instock_date'].dt.dayofweek
        df['weekofyear'] = df['instock_date'].dt.isocalendar().week.astype(int)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df.drop(columns='instock_date', inplace=True)

    for col in ['charges_1', 'charges_2 (%)', 'Minimum_price', 'Maximum_price']:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isnull().astype(int)
            df[col] = df[col].fillna(df[col].median())

    if 'Selling_Price' in df.columns:
        df = df[df['Selling_Price'] >= 0]
        df = df[df['Selling_Price'] < df['Selling_Price'].quantile(0.995)]

    if all(col in df.columns for col in ['Minimum_price', 'Maximum_price', 'charges_1']):
        df['price_range'] = df['Maximum_price'] - df['Minimum_price']
        df['price_ratio'] = df['charges_1'] / (df['Maximum_price'] + 1)
        df['log_charges_1'] = np.log1p(df['charges_1'])

    for col in ['Loyalty_customer', 'Discount_avail']:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes

    if 'Product_Category' in df.columns:
        df = pd.get_dummies(df, columns=['Product_Category'], drop_first=True)

    return df

def train_and_stack(train_df, test_df, test_ids):
    X = train_df.drop(columns='Selling_Price')
    y = train_df['Selling_Price']

    # Align test_df with X
    missing_cols = set(X.columns) - set(test_df.columns)
    for col in missing_cols:
        test_df[col] = 0
    test_df = test_df[X.columns]

    # Initialize models with GPU/CPU setting
    lgb_model = LGBMRegressor(n_estimators=1000, device='gpu' if USE_GPU else 'cpu')
    xgb_model = XGBRegressor(n_estimators=1000, tree_method='gpu_hist' if USE_GPU else 'auto', verbosity=0)
    cat_model = CatBoostRegressor(n_estimators=1000, task_type='GPU' if USE_GPU else 'CPU', verbose=0)

    models = [lgb_model, xgb_model, cat_model]
    oof_preds = np.zeros((len(X), len(models)))
    test_preds = np.zeros((len(test_df), len(models)))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"📦 Fold {fold + 1}")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        for i, model in enumerate(models):
            model.fit(X_train, y_train)
            oof_preds[val_idx, i] = model.predict(X_val)
            test_preds[:, i] += model.predict(test_df) / kf.n_splits

    meta_model = LinearRegression()
    meta_model.fit(oof_preds, y)
    final_preds = meta_model.predict(test_preds)

    joblib.dump({
        'lgb': lgb_model,
        'xgb': xgb_model,
        'cat': cat_model,
        'meta_model': meta_model
    }, "resources/stacked_model.pkl")
    print("✅ Saved model to resources/stacked_model.pkl")

    submission = pd.DataFrame({
        "Product_id": test_ids,
        "Selling_Price": final_preds
    })
    submission.to_csv("Dataset/submission.csv", index=False)
    print("✅ Dataset/submission.csv created!")

    return submission

# === RUN ===
if __name__ == "__main__":
    train = pd.read_csv("Dataset/train.csv")
    test = pd.read_csv("Dataset/test.csv")
    test_ids = test["Product_id"]

    train_clean = clean_and_engineer_features(train)
    test_clean = clean_and_engineer_features(test)

    submission = train_and_stack(train_clean, test_clean, test_ids)
