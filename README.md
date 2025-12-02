
# Delivery-Time Prediction with Horizon Models and Stacking  
*DSCI 631 Final Project*

This project develops a machine learning framework for predicting delivery times using a combination of a full-feature baseline model, specialized “horizon” models, and a linear stacking approach. The motivation is to evaluate whether models that specialize in different regions of the delivery-time distribution can produce improved and more stable performance compared to a traditional single-model architecture.

The structure is inspired by concepts described in DoorDash’s research on ETA prediction, adapted into a lightweight and interpretable form suitable for academic use.

---

## Project Overview

Accurate delivery-time prediction is critical for user experience and operational efficiency. Delivery systems fluctuate based on prep-time variability, driver availability, market load, and traffic patterns. Because these factors influence different deliveries in different ways, a single model often cannot capture the full complexity of the system.

This project implements and evaluates a modeling strategy built around:

- A full-feature **baseline LightGBM model**, serving as the primary benchmark.
- Three **horizon-specific models**, each specializing in a different segment of the delivery lifecycle:
  - **Short-Horizon (SH)**: models quick deliveries and early operational signals.
  - **Medium-Horizon (MH)**: focuses on dynamic marketplace telemetry.
  - **Long-Horizon (LH)**: emphasizes slower-moving structural patterns, such as store-level effects.
- A **RidgeCV stacking model** that blends predictions from all models using leakage-safe out-of-fold inputs.

The core question examined is whether horizon-specific specialization can produce more robust and consistent predictions than a single unified model.

---

## Dataset Description

The dataset contains approximately 197,000 delivery records with the following feature groups:

- **Order details**: item counts, subtotal values, price ranges.
- **Store attributes**: store IDs, primary category, protocol information.
- **Marketplace telemetry**: number of dashers on shift, number busy, outstanding orders.
- **Estimated durations**: DoorDash-generated prep and travel time estimates.
- **Timestamps**: converted to a consistent timezone for analysis.

The target variable is computed as the difference between the order creation timestamp and the final delivery timestamp, expressed in seconds.

This dataset is a simplified version of the telemetry used in real ETA systems and does not include the large-scale time-series and embedding-based features used in production.

---

## Project Architecture

Project/
│
├── data/
│ ├── raw/
│ ├── processed/
│ └── final/
│
├── notebooks/
│ ├── 01_data_clean_and_eda.ipynb
│ ├── 02_feature_engineering.ipynb
│ └── 03_modeling.ipynb
│
├── src/
│ ├── preprocessing.py
│ ├── feature_engineering.py
│ └── modeling.py
│ 
│
├── models/
│ ├── baseline_fold_.txt
│ ├── sh_fold_.txt
│ ├── mh_fold_.txt
│ ├── lh_fold_.txt
│ ├── final_stacking_model.pkl
│ └── stacking_weights.csv
│
└── reference documents/

---

## Modeling Approach

The workflow consists of three stages:

### 1. Baseline Model  
A single LightGBM model trained on the full engineered feature set. This provides the primary benchmark for evaluating alternative architectures.

### 2. Horizon Models  
Three LightGBM models trained on targeted feature subsets:

- **Short-Horizon Model (SH)**: emphasizes item-level features, prep-time estimates, and short-term telemetry.
- **Medium-Horizon Model (MH)**: focuses on real-time market load and supply-demand balance.
- **Long-Horizon Model (LH)**: prioritizes structural features such as store identity and high-level duration signals.

These models capture different behaviors within the delivery-time distribution.

### 3. Stacking Model  
A **RidgeCV regression stacker** blends predictions from all horizon models and the baseline.  
Key properties:

- Uses **out-of-fold predictions**, ensuring the stacker is trained without target leakage.
- Produces interpretable weights.
- Stabilizes extreme behavior seen in individual horizon models.

---

## Reproducibility

To reproduce the project results:

1. Run preprocessing and EDA through the provided notebooks:
- `01_data_clean_and_eda.ipynb`
- `02_feature_engineering.ipynb`

2. Train models and generate predictions:
- `03_modeling.ipynb`

3. Saved model artifacts appear in the `models/` directory.

If executed in order, the notebooks will generate all intermediate datasets and trained model files.

---

## Results Summary

- The baseline model performs strongly and captures most of the structure in the dataset.
- Horizon models capture specialized signals that the baseline smooths over.
- The RidgeCV stacker produces a **modest but consistent improvement** in MAE compared to the baseline.
- Visual diagnostics (lateness deciles, hourly MAE, cumulative error gap) show where each specialist model contributes value.
- The lightweight stacking approach demonstrates the same principle found in more advanced ETA systems: combining specialized models yields more stable predictions than relying on a single model.

---

## Limitations

- The stacker uses global weights and cannot adjust emphasis dynamically based on time of day, load conditions, or order characteristics.
- The dataset lacks richer telemetry, historical context, and sequence data used in production ETA systems.
- Feature interactions are limited to targeted engineering rather than learned encoders.
- The horizon definitions are static and could benefit from a more data-driven segmentation strategy.

---

## Future Work

Potential extensions include:

- Replacing the linear stacker with a non-linear meta-learner (gradient boosting, random forest, or neural network).
- Introducing context-aware weighting mechanisms to dynamically choose among horizon models.
- Adding richer interaction features or temporal attention mechanisms.
- Incorporating short-term sequential telemetry through lightweight encoders.

---

## References

- DoorDash Engineering Blog: *Precision in Motion: Deep Learning for Smarter ETA Predictions*
- LightGBM documentation
- Scikit-learn documentation
- Course materials for DSCI 631

---

