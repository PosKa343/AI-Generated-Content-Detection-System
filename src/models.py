import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


class AIDetector:

    def __init__(self, random_state=42):

        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.feature_names = None

    def prepare_data(self, X: np.ndarray, y: np.ndarray,
                     test_size: float = 0.2) -> Tuple:

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Class distribution in training:")
        print(f"  Human (0): {(y_train == 0).sum()}")
        print(f"  AI (1): {(y_train == 1).sum()}")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                                  regularization: str = 'l2', C: float = 1.0) -> LogisticRegression:

        print(
            f"\nTraining Logistic Regression (regularization={regularization}, C={C})...")

        if regularization == 'elasticnet':
            solver = 'saga'
            l1_ratio = 0.5  # Mix of L1 and L2
        else:
            solver = 'liblinear' if regularization == 'l1' else 'lbfgs'
            l1_ratio = None

        model = LogisticRegression(
            penalty=regularization,
            C=C,
            solver=solver,
            max_iter=1000,
            random_state=self.random_state,
            l1_ratio=l1_ratio
        )

        model.fit(X_train, y_train)

        self.models['logistic_regression'] = model
        print("Logistic Regression training complete")

        return model

    def get_feature_importance_lr(self, model: LogisticRegression,
                                  feature_names: List[str], top_n: int = 20):

        coefficients = model.coef_[0]

        top_positive_idx = np.argsort(coefficients)[-top_n:][::-1]
        top_negative_idx = np.argsort(coefficients)[:top_n]

        print(f"\nTop {top_n} features indicating AI-generated text:")
        for idx in top_positive_idx:
            print(f"  {feature_names[idx]}: {coefficients[idx]:.4f}")

        print(f"\nTop {top_n} features indicating human-written text:")
        for idx in top_negative_idx:
            print(f"  {feature_names[idx]}: {coefficients[idx]:.4f}")

        return coefficients

    def train_naive_bayes(self, X_train: np.ndarray, y_train: np.ndarray,
                          variant: str = 'gaussian') -> GaussianNB:

        print(f"\nTraining Naive Bayes ({variant})...")

        if variant == 'gaussian':
            model = GaussianNB()
        else:
            X_train = np.abs(X_train)
            model = MultinomialNB()

        model.fit(X_train, y_train)

        self.models['naive_bayes'] = model
        print("Naive Bayes training complete")

        return model

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                            n_estimators: int = 100, max_depth: int = None) -> RandomForestClassifier:

        print(f"\nTraining Random Forest (n_estimators={n_estimators})...")

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1  # Use all CPU cores
        )

        model.fit(X_train, y_train)

        self.models['random_forest'] = model
        print("Random Forest training complete")

        return model

    def get_feature_importance_rf(self, model: RandomForestClassifier,
                                  feature_names: List[str], top_n: int = 20):

        importances = model.feature_importances_

        top_idx = np.argsort(importances)[-top_n:][::-1]

        print(f"\nTop {top_n} most important features (Random Forest):")
        for idx in top_idx:
            print(f"  {feature_names[idx]}: {importances[idx]:.4f}")

        return importances

    def tune_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:

        print("\nTuning Logistic Regression hyperparameters...")

        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }

        grid_search = GridSearchCV(
            LogisticRegression(max_iter=1000, random_state=self.random_state),
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")

        self.best_model = grid_search.best_estimator_
        self.models['logistic_regression_tuned'] = grid_search.best_estimator_

        return grid_search.best_estimator_

    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray,
                             cv: int = 5) -> Dict[str, float]:

        print(f"\nPerforming {cv}-fold cross-validation...")

        scores = {
            'accuracy': cross_val_score(model, X, y, cv=cv, scoring='accuracy'),
            'precision': cross_val_score(model, X, y, cv=cv, scoring='precision'),
            'recall': cross_val_score(model, X, y, cv=cv, scoring='recall'),
            'f1': cross_val_score(model, X, y, cv=cv, scoring='f1')
        }

        print("\nCross-validation results:")
        for metric, values in scores.items():
            print(f"  {metric}: {values.mean():.4f} (+/- {values.std() * 2:.4f})")

        return scores

    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray,
                       model_name: str = "Model") -> Dict[str, float]:

        print(f"\nEvaluating {model_name}...")

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(
            model, 'predict_proba') else None

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }

        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)

        print(f"\n{model_name} Performance:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {cm[0, 0]}")
        print(f"  False Positives: {cm[0, 1]}")
        print(f"  False Negatives: {cm[1, 0]}")
        print(f"  True Positives:  {cm[1, 1]}")

        print(f"\nClassification Report:")
        print(classification_report(
            y_test, y_pred, target_names=['Human', 'AI']))

        return metrics, y_pred, y_pred_proba

    def compare_models(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:

        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)

        results = []

        for model_name, model in self.models.items():
            metrics, _, _ = self.evaluate_model(
                model, X_test, y_test, model_name)
            metrics['model'] = model_name
            results.append(metrics)

        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.set_index('model')

        print("\n" + "="*60)
        print("SUMMARY TABLE")
        print("="*60)
        print(comparison_df.to_string())

        return comparison_df

    def save_model(self, model, filepath: str):

        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):

        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
