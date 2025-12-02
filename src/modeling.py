# src/modeling.py
"""
LightGBM OOF Training and Ridge Stacking Utilities
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

def run_lgb_oof(
    full_df,            
    test_df,
    features, 
    TARGET,
    params_override=None,
    n_folds=5,          
    verbose=True
):
    # LightGBM K-Fold OOF Training
    X = full_df[features]
    y = full_df[TARGET]
    X_test = test_df[features]
    
    # Setup Storage
    oof_preds = np.zeros(len(full_df))
    test_preds_accum = np.zeros(len(test_df))
    models = []
    
    # Base Params
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 32,    
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "seed": 42,
        "verbosity": -1,
    }
    if params_override:
        params.update(params_override)

    # K-Fold Loop
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]
        
        dtrain = lgb.Dataset(X_train_fold, y_train_fold)
        dval = lgb.Dataset(X_val_fold, y_val_fold)
        
        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dtrain, dval],
            num_boost_round=2000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0) # Silence distinct fold logs
            ]
        )
        
        # OOF Prediction
        # The model never saw these rows during training or early stopping
        oof_preds[val_idx] = model.predict(X_val_fold)
        
        # Test Prediction (Average across folds)
        test_preds_accum += np.asarray(model.predict(X_test)).ravel() / n_folds
        models.append(model)
        
    # Metrics
    mae_oof = mean_absolute_error(y, oof_preds)
    rmse_oof = np.sqrt(mean_squared_error(y, oof_preds))
    
    if verbose:
        print(f"OOF MAE: {mae_oof:.2f}")
        print(f"OOF RMSE: {rmse_oof:.2f}")

    return oof_preds, test_preds_accum, models

def run_linear_stacking(valid_preds, test_preds, y_valid, y_test, verbose=True):
    """
    Robust Ridge Stacking with MAE/RMSE reporting.
    Recommended for production over LightGBM stacking when base models are correlated.
    """
    # Safety Check
    if set(valid_preds.keys()) != set(test_preds.keys()):
        raise ValueError(f"Mismatch in model keys! Valid: {list(valid_preds.keys())}, Test: {list(test_preds.keys())}")

    # Sort keys to ensure columns are in the same order for train/test
    sorted_keys = sorted(valid_preds.keys())
    
    X_train = pd.DataFrame({k: valid_preds[k] for k in sorted_keys})
    X_test = pd.DataFrame({k: test_preds[k] for k in sorted_keys})
    
    # Train RidgeCV (Optimizes MSE/RMSE implicitly)
    meta_model = RidgeCV(alphas=np.array([0.01, 0.1, 1.0, 10.0, 50.0]), scoring='neg_root_mean_squared_error')
    meta_model.fit(X_train, y_valid)
    
    # Predict
    pred_valid = meta_model.predict(X_train)
    pred_test = meta_model.predict(X_test)
    
    # Metrics
    mae_valid = mean_absolute_error(y_valid, pred_valid)
    rmse_valid = np.sqrt(mean_squared_error(y_valid, pred_valid))
    
    mae_test = mean_absolute_error(y_test, pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
    
    # Extract Weights
    weights = pd.Series(meta_model.coef_, index=X_train.columns, name="weight")
    intercept = meta_model.intercept_

    if verbose:
        print("\n==== Ridge Stacking Model ====")
        print(f"Selected Alpha: {meta_model.alpha_}")
        print(f"Intercept: {intercept:.4f}")
        print("Weights:")
        print(weights.sort_values(ascending=False))
        print("-" * 30)
        print(f"Stack Valid MAE:  {mae_valid:.4f}")
        print(f"Stack Valid RMSE: {rmse_valid:.4f}")
        print("-" * 30)
        print(f"Stack Test  MAE:  {mae_test:.4f}")
        print(f"Stack Test  RMSE: {rmse_test:.4f}")
        
    return meta_model, weights, mae_valid, rmse_valid