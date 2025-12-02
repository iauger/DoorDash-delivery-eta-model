# src/preprocessing_imputation.py
"""
Contains functions for leakage-safe preprocessing and imputation of missing values.
"""


import numpy as np
import pandas as pd
from tqdm import tqdm

TELEMETRY_COLS = [
        "total_onshift_dashers",
        "total_busy_dashers",
        "total_outstanding_orders"
    ]

PRICE_COLS = ["min_item_price", "max_item_price", "subtotal"]

def convert_timezone(df, time_column, from_tz = 'UTC', to_tz = 'US/Pacific'):
    """
    Convert the timezone of a datetime column in a DataFrame.
    """
    if time_column not in df.columns:
        raise ValueError(f"Column '{time_column}' does not exist in the DataFrame.")

    # Ensure datetime dtype
    df = df.copy()
    
    df[time_column] = pd.to_datetime(df[time_column], errors="coerce")

    if df[time_column].dt.tz is None:
        df[time_column] = df[time_column].dt.tz_localize(from_tz)

    df[time_column] = df[time_column].dt.tz_convert(to_tz)
    
    return df

def construct_temporal_features(df, time_column):
    """
    Construct temporal features from a datetime column in a DataFrame.
    """
    if time_column not in df.columns:
        raise ValueError(f"Column '{time_column}' does not exist in the DataFrame.")

    df = df.copy()
    
    df[time_column] = pd.to_datetime(df[time_column], errors="coerce")

    # Base time features
    df["time_hour"] = df[time_column].dt.hour
    df["time_day_of_week"] = df[time_column].dt.dayofweek
    df["time_month"] = df[time_column].dt.month
    df["time_is_weekend"] = df["time_day_of_week"].isin([5, 6]).astype(int)

    # Traffic-heavy periods (typical U.S. rush hour)
    df["time_is_rush_hour"] = df["time_hour"].isin([16, 17, 18]).astype(int)
    
    # Minute of day (0-1439)
    df["time_minute_of_day"] = df[time_column].dt.hour * 60 + df[time_column].dt.minute

    # 48 half-hour buckets
    df["time_bucket_48"] = df["time_minute_of_day"] // 30

    # Cyclical transforms (most useful for NN or smooth time modeling)
    df["time_hour_sin"] = np.sin(2 * np.pi * df["time_hour"] / 24)
    df["time_hour_cos"] = np.cos(2 * np.pi * df["time_hour"] / 24)
    df["time_dow_sin"] = np.sin(2 * np.pi * df["time_day_of_week"] / 7)
    df["time_dow_cos"] = np.cos(2 * np.pi * df["time_day_of_week"] / 7)
    
    # Rename base columns for clarity
    df = df.rename(columns={"estimated_order_place_duration": "time_estimated_order_place_duration",
                            "estimated_store_to_consumer_driving_duration": "time_estimated_store_to_consumer_driving_duration"})
    
    return df

def compute_delivery_time(df, order_time_col, delivery_time_col):
    """
    Compute delivery time in seconds between order and delivery timestamps.
    """
    if order_time_col not in df.columns:
        raise ValueError(f"Column '{order_time_col}' does not exist in the DataFrame.")
    if delivery_time_col not in df.columns:
        raise ValueError(f"Column '{delivery_time_col}' does not exist in the DataFrame.")
    
    df = df.copy()
    df[order_time_col] = pd.to_datetime(df[order_time_col], errors="coerce")
    df[delivery_time_col] = pd.to_datetime(df[delivery_time_col], errors="coerce")

    df["target_delivery_seconds"] = (
        df[delivery_time_col] - df[order_time_col]
    ).dt.total_seconds()

    # Remove invalid rows
    df = df.dropna(subset=["target_delivery_seconds"])
    df = df[df["target_delivery_seconds"] >= 0]

    return df

def impute_actual_delivery_time(df):
    """
    Remove rows with missing actual delivery time.
    """
    df = df.copy()
    df = df.dropna(subset=["actual_delivery_time"])
    return df

def hierarchical_median_impute(df, target_col, group_cols):
    """
    Impute missing values in `target_col` using hierarchical medians.
    Hierarchy proceeds from group_cols[0] to group_cols[1] to global median.
    """
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
        
    for col in group_cols:
        if col not in df.columns:
            raise ValueError(f"Grouping column '{col}' not found in DataFrame.")

    group_medians = []
    for col in group_cols:
        median_map = df.groupby(col, observed=True)[target_col].median().to_dict()
        group_medians.append((col, median_map))

    global_median = df[target_col].median()

    for col, median_map in group_medians:
        missing_mask = df[target_col].isna()
        df.loc[missing_mask, target_col] = (
            df.loc[missing_mask].apply(
                lambda row: median_map.get(row[col], np.nan),
                axis=1
            )
        )

    missing_mask = df[target_col].isna()
    df.loc[missing_mask, target_col] = global_median

    if pd.api.types.is_integer_dtype(df[target_col].dtype):
        df[target_col] = df[target_col].round().astype("Float32")

    return df

def hierarchical_categorical_impute(df, target_col, group_col, default=None):
    """
    Impute missing categorical values in `target_col` by using the most common 
    non-null value within each `group_col` group. Falls back to `default`.
    """
    if default is None:
        raise ValueError("Default value must be explicitly provided.")

    df = df.copy()

    if isinstance(df[target_col].dtype, pd.CategoricalDtype):
        df[target_col] = df[target_col].cat.add_categories([default])

    mapping = (
        df.groupby(group_col, observed=True)[target_col]
          .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else default)
          .to_dict()
    )

    missing_mask = df[target_col].isna()

    df.loc[missing_mask, target_col] = (
        df.loc[missing_mask, group_col]
          .map(mapping)
          .fillna(default)
    )
    return df

def impute_market_id(df):
    
    result = hierarchical_categorical_impute(df, "market_id", "store_id", default="unknown")
    return result

def impute_store_primary_category(df):
    
    result = hierarchical_categorical_impute(df, "store_primary_category", "store_id", default="unknown")
    return result

def impute_order_protocol(df):
    df = df.copy()
    
    df["order_protocol"] = df["order_protocol"].fillna("unknown")
    return df

def impute_estimated_duration(df):
    df = hierarchical_median_impute(
        df,
        "time_estimated_store_to_consumer_driving_duration",
        ["store_id", "market_id"]
    )
    df = hierarchical_median_impute(
        df,
        "time_estimated_order_place_duration",
        ["store_id", "market_id"]
    )
    return df
    
def impute_marketplace_telemetry(df):
    df = df.copy()
    df["load_telemetry_missing"] = df[TELEMETRY_COLS].isna().all(axis=1).astype(int)

    # Clip negative values
    df[TELEMETRY_COLS] = df[TELEMETRY_COLS].clip(lower=0)

    # Hierarchical numeric imputation
    for col in TELEMETRY_COLS:
        df = hierarchical_median_impute(df, col, ["store_id", "market_id"])
    return df

def compute_item_metrics(df):
    """
    Compute additional item-level metrics.
    """
    df = df.copy()

    df[PRICE_COLS] = df[PRICE_COLS].clip(lower=0)
    
    # Rename base order columns for clarity
    df = df.rename(columns={"total_items": "order_total_items",
                            "subtotal": "order_subtotal",
                            "max_item_price": "order_max_item_price",
                            "min_item_price": "order_min_item_price",
                            "num_distinct_items": "order_num_distinct_items"})
    
    df["order_avg_item_price"] = df["order_subtotal"] / df["order_total_items"].replace(0,1)
    df["order_item_price_range"] = df["order_max_item_price"] - df["order_min_item_price"]
    df["order_percent_distinct_items"] = df["order_num_distinct_items"].replace(0,1) / df["order_total_items"]
    
    store_medians = df.groupby("store_id", observed=True)["order_subtotal"].transform("median")
    df["order_high_value"] = (df["order_subtotal"] > store_medians).astype(int)
    
    # Bucketize key numeric features for non-linear effects and to reduce outlier impact
    df["order_subtotal_bucket"] = pd.qcut(
        df["order_subtotal"], q=4, labels=False, duplicates="drop"
    )

    df["order_total_items_bucket"] = pd.cut(
        df["order_total_items"], bins=[0, 1, 3, 5, 10, np.inf], labels=False
    )

    df["order_distinct_items_bucket"] = pd.qcut(
        df["order_num_distinct_items"], q=4, labels=False, duplicates="drop"
    )
    
    return df

def compute_store_features(df):
    """
    Compute safe, non-leaky store-level features using only model-provided estimated durations.
    """
    df = df.copy()
    store_means = df.groupby("store_id", observed=True)[
        ["time_estimated_order_place_duration",
         "time_estimated_store_to_consumer_driving_duration"]
    ].transform("mean")

    df["store_avg_order_place_duration"] = store_means["time_estimated_order_place_duration"]
    df["store_avg_drive_duration"] = store_means["time_estimated_store_to_consumer_driving_duration"]
    df["store_order_volume"] = df.groupby("store_id", observed=True)["store_id"].transform("count")

    return df

def compute_marketplace_load(df, window=10):
    """
    Construct marketplace load features using rolling averages to capture supply, demand and congestion.
    """
    df = df.copy()
    # Clean telemetry columns defensively, should be clean already
    df[TELEMETRY_COLS] = df[TELEMETRY_COLS].clip(lower=0) 
    
    safe_onshift = df["total_onshift_dashers"].replace(0, np.nan)
    safe_busy = df["total_busy_dashers"].replace(0, np.nan)
    
    # Key ratios
    df["load_busy_ratio"] = (df["total_busy_dashers"] / safe_onshift)
    df["load_demand_supply_ratio"] = (df["total_outstanding_orders"] / safe_onshift)
    df["load_orders_per_busy_dasher"] = (df["total_outstanding_orders"] / safe_busy)
    
    df[["load_busy_ratio", "load_demand_supply_ratio", "load_orders_per_busy_dasher"]] = \
        df[["load_busy_ratio", "load_demand_supply_ratio", "load_orders_per_busy_dasher"]].fillna(0)
        
    # Rate of change / velocity features
    df = df.sort_values(["market_id", "created_at"])

    # Helper: rolling apply over each market_id
    def roll_mean(col):
        return (
            df.groupby("market_id", observed=True)[col]
              .rolling(window=window, min_periods=1)
              .mean()
              .reset_index(level=0, drop=True)
        )

    # Rolling standard deviation
    def roll_std(col):
        return (
            df.groupby("market_id", observed=True)[col]
              .rolling(window=window, min_periods=1)
              .std()
              .reset_index(level=0, drop=True)
        )
    
    # Rolling features
    df["load_roll_outstanding_mean"]   = roll_mean("total_outstanding_orders")
    df["load_roll_onshift_mean"]       = roll_mean("total_onshift_dashers")
    df["load_roll_busy_mean"]          = roll_mean("total_busy_dashers")
    df["load_roll_busy_ratio_mean"]    = roll_mean("load_busy_ratio")
    df["load_roll_demand_supply_mean"] = roll_mean("load_demand_supply_ratio")
    
    df["load_roll_outstanding_std"] = roll_std("total_outstanding_orders")
    df["load_roll_busy_std"]        = roll_std("total_busy_dashers")
    
    df[["load_roll_busy_std", "load_roll_outstanding_std"]] = \
        df[["load_roll_busy_std", "load_roll_outstanding_std"]].fillna(0)
    
    df["load_recent_order_count"] = (
        df.groupby("market_id", observed=True)["created_at"]
          .rolling(window=window, min_periods=1)
          .count()
          .reset_index(level=0, drop=True)
    )
    
    # Momentum features
    df["load_demand_momentum"] = (
        df.groupby("market_id", observed=True)["total_outstanding_orders"].diff().fillna(0)
    )
    df["load_supply_momentum"] = (
        df.groupby("market_id", observed=True)["total_onshift_dashers"].diff().fillna(0)
    )
    df["load_busy_momentum"] = (
        df.groupby("market_id", observed=True)["total_busy_dashers"].diff().fillna(0)
    )

    # Smooth momentum with rolling mean
    df["load_demand_momentum"] = df.groupby("market_id", observed=True)["load_demand_momentum"].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    df["load_supply_momentum"] = df.groupby("market_id", observed=True)["load_supply_momentum"].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    df["load_busy_momentum"]   = df.groupby("market_id", observed=True)["load_busy_momentum"].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

    # Composite indices
    # Stress Index — undersupply indicator
    df["load_supply_stress_index"] = (
        df["load_busy_ratio"] * df["load_demand_supply_ratio"]
    ).fillna(0)

    # Volatility Index — how unstable the market is
    df["load_utilization_volatility"] = (
        df["load_roll_busy_std"] / (df["load_roll_busy_mean"] + 1)
    ).fillna(0)

    # Pressure Index — "bad things are happening" indicator
    df["load_pressure_index"] = (
        df["total_outstanding_orders"] * df["load_utilization_volatility"]
    ).fillna(0)

    return df

def compute_lagged_telemetry(df):
    """
    Create lagged and short-term volatility telemetry features.
    Based on DoorDash ETA model design: capture short-term market dynamics.
    """
    df = df.copy()

    # Make sure created_at is a clean datetime and drop any NaT rows
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["created_at"])
    if len(df) != before:
        print(f"Dropped {before - len(df)} rows with NaT created_at before lagged telemetry.")

    tele_cols = [
        "total_onshift_dashers",
        "total_busy_dashers",
        "total_outstanding_orders",
        "load_busy_ratio",
        "load_demand_supply_ratio",
    ]

    lag_specs = [
        ("5min",  pd.Timedelta("5min")),
        ("15min", pd.Timedelta("15min")),
        ("30min", pd.Timedelta("30min")),
    ]

    # Pre-create lag columns as float so we can assign into them
    for col in tele_cols:
        for label, _ in lag_specs:
            df[f"{col}_lag_{label}"] = np.nan

    # ---- Per-market processing ------------------------------------
    for mkt, idx in df.groupby("market_id").groups.items():
        g = df.loc[idx].sort_values("created_at")

        # For each telemetry column, compute time-lagged values
        for col in tele_cols:
            left = g[["created_at"]]                 # times we want lagged values for
            right = g[["created_at", col]]          # history of that telemetry

            for label, delta in lag_specs:
                merged = pd.merge_asof(
                    left.sort_values("created_at"),
                    right.sort_values("created_at"),
                    on="created_at",
                    direction="backward",
                    tolerance=delta,
                )
                df.loc[g.index, f"{col}_lag_{label}"] = merged[col].to_numpy()

    # ---- Rolling volatility (order-based) -------------------------
    for col in tele_cols:
        df[f"{col}_roll10_std"] = (
            df.groupby("market_id", observed=True)[col]
              .rolling(window=10, min_periods=5)
              .std()
              .reset_index(level=0, drop=True)
        )

    # Bursty features
    df["demand_spike_15m"] = (
        df["total_outstanding_orders"]
        - df["total_outstanding_orders_lag_15min"]
    )

    df["supply_drop_15m"] = (
        df["total_onshift_dashers"]
        - df["total_onshift_dashers_lag_15min"]
    )

    df["busy_growth_15m"] = (
        df["total_busy_dashers"]
        - df["total_busy_dashers_lag_15min"]
    )

    df["burst_index"] = (
        df["demand_spike_15m"] - df["supply_drop_15m"]
    ).clip(lower=0)

    # Identify only telemetry-derived columns
    lag_cols = [
        col for col in df.columns
        if ("lag_" in col)
        or ("roll10_std" in col)
        or (col in ["demand_spike_15m", "supply_drop_15m", "busy_growth_15m", "burst_index"])
    ]

    df[lag_cols] = df[lag_cols].fillna(0)

    return df

def compute_velocity_features(df):
    """
    Captures the rate of change in market conditions (Velocity).
    """
    df = df.copy()
    
    # Ratio of 5min (Short term) to 30min (Medium term) load
    # > 1.0 means market is heating up (Velocity Positive)
    # < 1.0 means market is cooling down (Velocity Negative)
    
    # Safe division
    df["market_velocity_orders"] = (
        df["total_outstanding_orders_lag_5min"] / 
        (df["total_outstanding_orders_lag_30min"] + 1)
    )
    
    df["market_velocity_busy"] = (
        df["total_busy_dashers_lag_5min"] / 
        (df["total_busy_dashers_lag_30min"] + 1)
    )
    
    return df

def compute_robust_buckets(df):
    """
    Discretizes continuous features to handle outliers.
    """
    df = df.copy()
    
    # Quantile Buckets (Robust to outliers)
    # 0 = Low, 1 = Med-Low, 2 = Med-High, 3 = High
    df["bucket_distance"] = pd.qcut(df["time_estimated_store_to_consumer_driving_duration"], q=4, labels=False, duplicates='drop')
    df["bucket_dashers"] = pd.qcut(df["total_onshift_dashers"], q=4, labels=False, duplicates='drop')
    
    return df

def filter_target_outliers_iqr(df, column, multiplier=3.0):
    """
    Removes extreme outliers from the target column using an adjustable IQR rule.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    upper = Q3 + multiplier * IQR
    lower = max(Q1 - multiplier * IQR, 0)   

    print(f"IQR filtering:\n  Q1={Q1:,.2f}, Q3={Q3:,.2f}, IQR={IQR:,.2f}")
    print(f"  lower bound={lower:,.2f}, upper bound={upper:,.2f}")
    print(f"  Rows before: {len(df):,}")

    df_filtered = df[(df[column] >= lower) & (df[column] <= upper)]
    
    print(f"  Rows after: {len(df_filtered):,}  (removed {len(df) - len(df_filtered):,})")

    return df_filtered

def preprocess_impute(df, verbose=False):
    """
    The only function that pipelines ALL leakage-safe preprocessing.
    """
    df = df.copy()

    steps = [
        ("convert_timezone_created_at", lambda d: convert_timezone(d, "created_at")),
        ("convert_timezone_actual_delivery_time", lambda d: convert_timezone(d, "actual_delivery_time")),
        ("construct_temporal_features", lambda d: construct_temporal_features(d, "created_at")),
        ("impute_actual_delivery_time", impute_actual_delivery_time),
        ("impute_market_id", impute_market_id),
        ("impute_store_primary_category", impute_store_primary_category),
        ("impute_order_protocol", impute_order_protocol),
        ("impute_estimated_duration", impute_estimated_duration),
        ("impute_marketplace_telemetry", impute_marketplace_telemetry),
        ("compute_delivery_time", lambda d: compute_delivery_time(d, "created_at", "actual_delivery_time")),
        ("filter_target_delivery_outliers", lambda d: filter_target_outliers_iqr(d, "target_delivery_seconds", multiplier=3.0)),
        ("compute_item_metrics", compute_item_metrics),
        ("filter_item_count_outliers", lambda d: filter_target_outliers_iqr(d, "order_total_items", multiplier=15.0)),
        ("compute_store_features", compute_store_features),
        ("compute_marketplace_load", compute_marketplace_load),
        ("compute_lagged_telemetry", compute_lagged_telemetry),
        ("compute_velocity_features", compute_velocity_features),
        ("compute_robust_buckets", compute_robust_buckets),
    ]

    print("Starting preprocessing and imputation pipeline...")

    for name, func in tqdm(steps, desc="Preprocessing Pipeline", unit="step"):
        if verbose:
            print(f"Running: {name}")
        df = func(df)

    print("Preprocessing and imputation pipeline complete.")
    return df
