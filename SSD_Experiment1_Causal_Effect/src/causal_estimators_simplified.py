#!/usr/bin/env python3
"""
Simplified causal estimators for imputed data analysis
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_predict
import warnings

def run_tmle_simple(df, outcome_col, treatment_col, covariate_cols):
    """Simplified TMLE implementation"""
    try:
        Y = df[outcome_col].values
        A = df[treatment_col].values
        W = df[covariate_cols].values
        
        # Handle missing values
        valid_idx = ~(np.isnan(Y) | np.isnan(A) | np.any(np.isnan(W), axis=1))
        Y = Y[valid_idx]
        A = A[valid_idx]
        W = W[valid_idx]
        
        # Outcome model
        outcome_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        Y_pred = cross_val_predict(outcome_model, np.column_stack([A, W]), Y, cv=3)
        outcome_model.fit(np.column_stack([A, W]), Y)
        
        # Propensity score
        ps_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        ps = cross_val_predict(ps_model, W, A, cv=3, method='predict_proba')[:, 1]
        
        # Bound propensity scores
        ps = np.clip(ps, 0.01, 0.99)
        
        # Calculate ATE
        n = len(Y)
        Q1 = outcome_model.predict(np.column_stack([np.ones(n), W]))
        Q0 = outcome_model.predict(np.column_stack([np.zeros(n), W]))
        
        ate = np.mean(Q1 - Q0)
        
        # Simple SE
        se = np.std(Q1 - Q0) / np.sqrt(n)
        
        return {
            'method': 'TMLE',
            'estimate': float(ate),
            'se': float(se),
            'ci_lower': float(ate - 1.96 * se),
            'ci_upper': float(ate + 1.96 * se),
            'n': int(n)
        }
    except Exception as e:
        return {
            'method': 'TMLE',
            'estimate': None,
            'error': str(e)
        }

def run_dml_simple(df, outcome_col, treatment_col, covariate_cols):
    """Simplified Double ML implementation"""
    try:
        Y = df[outcome_col].values
        T = df[treatment_col].values
        X = df[covariate_cols].values
        
        # Handle missing values
        valid_idx = ~(np.isnan(Y) | np.isnan(T) | np.any(np.isnan(X), axis=1))
        Y = Y[valid_idx]
        T = T[valid_idx]
        X = X[valid_idx]
        
        # Nuisance functions
        rf_y = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        rf_t = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        
        # Cross-fitting
        Y_res = Y - cross_val_predict(rf_y, X, Y, cv=3)
        T_res = T - cross_val_predict(rf_t, X, T, cv=3, method='predict_proba')[:, 1]
        
        # Final stage
        ate = np.sum(T_res * Y_res) / np.sum(T_res * T_res)
        
        # SE
        residuals = Y_res - ate * T_res
        n = len(Y)
        se = np.sqrt(np.sum(residuals**2) / (n - 1)) / np.sqrt(np.sum(T_res**2))
        
        return {
            'method': 'Double ML',
            'estimate': float(ate),
            'se': float(se),
            'ci_lower': float(ate - 1.96 * se),
            'ci_upper': float(ate + 1.96 * se),
            'n': int(n)
        }
    except Exception as e:
        return {
            'method': 'Double ML',
            'estimate': None,
            'error': str(e)
        }

def run_causal_forest_simple(df, outcome_col, treatment_col, covariate_cols):
    """Simplified Causal Forest implementation"""
    try:
        Y = df[outcome_col].values
        T = df[treatment_col].values
        X = df[covariate_cols].values
        
        # Handle missing values
        valid_idx = ~(np.isnan(Y) | np.isnan(T) | np.any(np.isnan(X), axis=1))
        Y = Y[valid_idx]
        T = T[valid_idx]
        X = X[valid_idx]
        
        # Separate forests for treated/control
        treated_idx = T == 1
        control_idx = T == 0
        
        rf_treated = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        rf_control = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        
        rf_treated.fit(X[treated_idx], Y[treated_idx])
        rf_control.fit(X[control_idx], Y[control_idx])
        
        # CATE
        Y1_pred = rf_treated.predict(X)
        Y0_pred = rf_control.predict(X)
        cate = Y1_pred - Y0_pred
        ate = np.mean(cate)
        
        # Bootstrap CI
        n_boot = 100
        ate_boot = []
        rng = np.random.RandomState(42)
        
        for _ in range(n_boot):
            idx = rng.choice(len(cate), size=len(cate), replace=True)
            ate_boot.append(cate[idx].mean())
        
        ci_lower = np.percentile(ate_boot, 2.5)
        ci_upper = np.percentile(ate_boot, 97.5)
        
        return {
            'method': 'Causal Forest',
            'estimate': float(ate),
            'se': float(np.std(ate_boot)),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n': int(len(Y))
        }
    except Exception as e:
        return {
            'method': 'Causal Forest',
            'estimate': None,
            'error': str(e)
        }
