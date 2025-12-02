"""
feature_engineering.py
This module contains classes and functions for feature engineering,
including leakage-safe target encoding and interaction feature creation.
"""

import numpy as np
from sklearn.model_selection import KFold


# CATEGORY TARGET ENCODER (Leakage-safe)

class TargetEncoderOOF:
    """
    K-Fold Target Encoding.
    For each fold, we calculate the mean of the OTHER folds to map the current fold.
    This prevents the model from seeing its own label in the feature.
    """
    def __init__(self, col, target_col, n_folds=5, smoothing=10):
        self.col = col
        self.target_col = target_col
        self.n_folds = n_folds
        self.smoothing = smoothing
        self.global_mean = None
        self.mapping_final = None # Used for Test/Inference

    def fit(self, df):
        self.global_mean = df[self.target_col].mean()
        
        # We also learn the "Final" mapping on the whole train set 
        # to be used ONLY on the Test set later.
        stats = df.groupby(self.col, observed=True)[self.target_col].agg(["mean", "count"])
        smoother = stats["count"] / (stats["count"] + self.smoothing)
        stats["encoded"] = smoother * stats["mean"] + (1 - smoother) * self.global_mean
        self.mapping_final = stats["encoded"].to_dict()
        return self

    def transform(self, df, is_training=False):
        df = df.copy()
        
        # Training (Generate OOF features to prevent leakage)
        if is_training:
            oof_values = np.full(len(df), np.nan)
            
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            
            for train_idx, val_idx in kf.split(df):
                X_tr, X_val = df.iloc[train_idx], df.iloc[val_idx]
                
                # Compute means on TRAIN fold
                means = X_tr.groupby(self.col, observed=True)[self.target_col].mean()
                
                # Map values and extract numpy array immediately
                mapped_values = X_val[self.col].map(means).values
                
                # Assign using integer positions 
                oof_values[val_idx] = mapped_values
            
            # Assign the full array to the DataFrame at once
            df[f"{self.col}_encoded"] = oof_values
            
            # 3. Fill NaNs (new stores/categories) with global mean
            df[f"{self.col}_encoded"] = df[f"{self.col}_encoded"].fillna(self.global_mean)
            
        # Inference (Use the full-train mapping)
        else:
            df[f"{self.col}_encoded"] = (
                df[self.col].map(self.mapping_final).fillna(self.global_mean)
            )
            
        return df

# INTERACTION FEATURES

class FeatureInteractionBuilder:
    """
    Creates hand-selected interaction terms that mimic CrossNet
    without neural modeling.
    """

    def __init__(self, interactions=None):
        """
        interactions = list of tuples:
            [("order_subtotal", "order_total_items"),
             ("time_hour", "category_target_encoded"), ...]
        """
        self.interactions = interactions or []

    def transform(self, df):
        df = df.copy()

        for a, b in self.interactions:
            df[f"interaction__{a}__x__{b}"] = df[a] * df[b]

        return df


