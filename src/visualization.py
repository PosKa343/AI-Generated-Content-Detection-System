import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.metrics import confusion_matrix, roc_curve, auc

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class Visualizer:

    def __init__(self, save_dir: str = None):

        self.save_dir = save_dir

        if self.save_dir:
            import os
            os.makedirs(self.save_dir, exist_ok=True)

    def _save_fig(self, filename: str):

        if self.save_dir:
            filepath = f"{self.save_dir}/{filename}"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {filepath}")

    def plot_class_distribution(self, labels: np.ndarray, class_names: List[str] = None):

        if class_names is None:
            class_names = ['Human', 'AI']

        plt.figure(figsize=(8, 6))

        unique, counts = np.unique(labels, return_counts=True)

        colors = ['#3498db', '#e74c3c']
        plt.bar(class_names, counts, color=colors,
                alpha=0.7, edgecolor='black')
        plt.ylabel('Count', fontsize=12)
        plt.title('Class Distribution in Dataset',
                  fontsize=14, fontweight='bold')

        for i, count in enumerate(counts):
            plt.text(i, count, str(count), ha='center',
                     va='bottom', fontsize=12, fontweight='bold')

        plt.tight_layout()
        self._save_fig('class_distribution.png')
        plt.show()

    def plot_feature_distributions(self, df: pd.DataFrame, features: List[str],
                                   label_column: str = 'label'):

        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_features > 1 else [axes]

        for idx, feature in enumerate(features):
            ax = axes[idx]

            for label, color, name in zip([0, 1], ['#3498db', '#e74c3c'], ['Human', 'AI']):
                data = df[df[label_column] == label][feature]
                ax.hist(data, bins=30, alpha=0.6, color=color,
                        label=name, edgecolor='black')

            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'Distribution of {feature}',
                         fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        self._save_fig('feature_distributions.png')
        plt.show()

    def plot_correlation_heatmap(self, df: pd.DataFrame, features: List[str]):

        plt.figure(figsize=(14, 12))

        corr_matrix = df[features].corr()

        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})

        plt.title('Feature Correlation Heatmap',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_fig('correlation_heatmap.png')
        plt.show()

    def plot_feature_comparison(self, df: pd.DataFrame, feature1: str, feature2: str,
                                label_column: str = 'label'):

        plt.figure(figsize=(10, 6))

        for label, color, name in zip([0, 1], ['#3498db', '#e74c3c'], ['Human', 'AI']):
            mask = df[label_column] == label
            plt.scatter(df[mask][feature1], df[mask][feature2],
                        alpha=0.6, s=30, color=color, label=name, edgecolors='black', linewidth=0.5)

        plt.xlabel(feature1, fontsize=12)
        plt.ylabel(feature2, fontsize=12)
        plt.title(f'{feature1} vs {feature2}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save_fig(f'comparison_{feature1}_vs_{feature2}.png')
        plt.show()

    def plot_boxplots(self, df: pd.DataFrame, features: List[str], label_column: str = 'label'):

        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_features > 1 else [axes]

        for idx, feature in enumerate(features):
            ax = axes[idx]

            data_to_plot = [df[df[label_column] == 0][feature].dropna(),
                            df[df[label_column] == 1][feature].dropna()]

            bp = ax.boxplot(data_to_plot, labels=[
                            'Human', 'AI'], patch_artist=True)

            colors = ['#3498db', '#e74c3c']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            ax.set_ylabel(feature, fontsize=10)
            ax.set_title(f'Boxplot of {feature}',
                         fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        self._save_fig('feature_boxplots.png')
        plt.show()

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              class_names: List[str] = None):

        if class_names is None:
            class_names = ['Human', 'AI']

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'}, square=True)

        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_fig('confusion_matrix.png')
        plt.show()

    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                       model_name: str = "Model"):

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='#e74c3c', lw=2,
                 label=f'{model_name} (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=1,
                 linestyle='--', label='Random Classifier')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save_fig(f'roc_curve_{model_name}.png')
        plt.show()

    def plot_feature_importance(self, importances: np.ndarray, feature_names: List[str],
                                top_n: int = 20, title: str = "Feature Importance"):

        # Get top N features
        top_idx = np.argsort(importances)[-top_n:]
        top_importances = importances[top_idx]
        top_names = [feature_names[i] for i in top_idx]

        plt.figure(figsize=(10, 8))
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))
        plt.barh(range(top_n), top_importances,
                 color=colors, edgecolor='black')
        plt.yticks(range(top_n), top_names)
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        self._save_fig('feature_importance.png')
        plt.show()

    def plot_model_comparison(self, comparison_df: pd.DataFrame):

        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        available_metrics = [m for m in metrics if m in comparison_df.columns]

        fig, axes = plt.subplots(1, len(available_metrics), figsize=(15, 5))

        if len(available_metrics) == 1:
            axes = [axes]

        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]

            values = comparison_df[metric].values
            models = comparison_df.index.tolist()

            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
            bars = ax.bar(range(len(models)), values,
                          color=colors, edgecolor='black', alpha=0.7)

            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison',
                         fontsize=12, fontweight='bold')
            ax.set_ylim([0, 1.0])
            ax.grid(True, alpha=0.3, axis='y')

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        self._save_fig('model_comparison.png')
        plt.show()
