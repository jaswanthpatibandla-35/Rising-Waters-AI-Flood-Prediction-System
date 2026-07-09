import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.decomposition import PCA

# 14 Classifiers Imports
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              AdaBoostClassifier, ExtraTreesClassifier, StackingClassifier, VotingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

logger = logging.getLogger("evaluate_models")

# Dynamic Imports for optional packages
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    logger.warning("LightGBM package not installed. Skipping LGBM during training.")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    logger.warning("CatBoost package not installed. Skipping CatBoost during training.")

class ModelEvaluator:
    def __init__(self, static_img_dir):
        self.static_img_dir = static_img_dir
        os.makedirs(static_img_dir, exist_ok=True)
        
    def get_all_models(self):
        """Initializes dictionary of models, checking for package installations."""
        base_models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
            'SVM': SVC(probability=True, kernel='rbf', random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
            'Extra Trees': ExtraTreesClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
            'XGBoost': XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='logloss')
        }

        if HAS_LGBM:
            base_models['LightGBM'] = LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1)
        if HAS_CATBOOST:
            base_models['CatBoost'] = CatBoostClassifier(iterations=100, depth=4, learning_rate=0.1, random_state=42, verbose=0)
            
        return base_models

    def evaluate_all(self, X_train, X_test, y_train, y_test):
        """Trains and compares all base classifiers, plus Stacking & Voting ensembles."""
        logger.info("Initializing 14-Classifier evaluation pipeline...")
        base_models = self.get_all_models()
        results = {}

        # 1. Evaluate Base Models
        for name, model in base_models.items():
            logger.info(f"Training and evaluating: {name}...")
            model.fit(X_train, y_train)
            results[name] = self._get_model_metrics(model, X_train, X_test, y_train, y_test)

        # 2. Build Stacking & Voting Classifiers using top estimators
        # Sort models based on F1-score to pick top 3 for stacking/voting
        sorted_models = sorted(results.items(), key=lambda item: item[1]['F1-score'], reverse=True)
        top_names = [m[0] for m in sorted_models[:3]]
        logger.info(f"Top 3 estimators selected for ensemble: {top_names}")

        estimators = [(name, base_models[name]) for name in top_names]

        # Stacking Classifier
        logger.info("Training Stacking Classifier...")
        stack_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=3,
            n_jobs=-1
        )
        stack_clf.fit(X_train, y_train)
        results['Stacking Classifier'] = self._get_model_metrics(stack_clf, X_train, X_test, y_train, y_test)

        # Voting Classifier
        logger.info("Training Voting Classifier...")
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        voting_clf.fit(X_train, y_train)
        results['Voting Classifier'] = self._get_model_metrics(voting_clf, X_train, X_test, y_train, y_test)

        # Re-include ensemble objects in a local dictionary to return
        all_models = {**base_models, 'Stacking Classifier': stack_clf, 'Voting Classifier': voting_clf}
        
        # 3. Choose best model based on F1-score
        best_model_name = max(results, key=lambda name: results[name]['F1-score'])
        best_model_obj = all_models[best_model_name]
        logger.info(f"Optimal Model Selected: {best_model_name} with F1: {results[best_model_name]['F1-score']:.4f}")

        # 4. Generate visual evaluation charts
        self.generate_plots(all_models, results, best_model_name, X_train, X_test, y_train, y_test)

        return results, best_model_name, best_model_obj

    def _get_model_metrics(self, model, X_train, X_test, y_train, y_test):
        """Evaluate a single model on train & test sets."""
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
        if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
            y_prob = y_prob[:, 1]
            
        # Standard classification scores
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)

        # 3-Fold Cross Validation Score
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1', n_jobs=-1)
        
        return {
            'Accuracy': float(acc),
            'Precision': float(prec),
            'Recall': float(rec),
            'F1-score': float(f1),
            'ROC-AUC': float(roc_auc),
            'CV-F1-Mean': float(np.mean(cv_scores))
        }

    def generate_plots(self, models, results, best_model_name, X_train, X_test, y_train, y_test):
        """Generate and save all assessment visuals to the static/images folder."""
        sns.set_theme(style="darkgrid")
        
        # 1. Bar Chart Comparison
        plt.figure(figsize=(14, 7))
        comp_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})
        comp_melted = pd.melt(comp_df, id_vars='Model', value_vars=['Accuracy', 'F1-score', 'ROC-AUC', 'CV-F1-Mean'],
                             var_name='Metric', value_name='Value')
        sns.barplot(data=comp_melted, x='Model', y='Value', hue='Metric', palette='viridis')
        plt.xticks(rotation=25, ha='right', fontsize=10)
        plt.ylim([0, 1.05])
        plt.title('Algorithm Evaluation Grid (14 Models Compared)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.static_img_dir, "accuracy_comparison.png"), dpi=150)
        plt.close()

        # 2. Multi-Model ROC Curves
        plt.figure(figsize=(10, 8))
        for name, model in models.items():
            if name in ['Stacking Classifier', 'Voting Classifier', best_model_name] or len(models) <= 5:
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc_score = roc_auc_score(y_test, y_prob)
                plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})", lw=2)
        plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Analysis', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.static_img_dir, "roc_curve.png"), dpi=150)
        plt.close()

        # 3. Precision-Recall Curve (Best Model)
        best_model = models[best_model_name]
        y_prob_best = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else best_model.decision_function(X_test)
        prec, rec, _ = precision_recall_curve(y_test, y_prob_best)
        plt.figure(figsize=(8, 6))
        plt.plot(rec, prec, color='purple', lw=2, label=f'{best_model_name}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve ({best_model_name})', fontsize=12, fontweight='bold')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.static_img_dir, "precision_recall_curve.png"), dpi=150)
        plt.close()

        # 4. Confusion Matrix (Best Model)
        y_pred_best = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_best)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['No Flood', 'Flood'], yticklabels=['No Flood', 'Flood'],
                    annot_kws={'size': 14, 'weight': 'bold'})
        plt.title(f'Confusion Matrix | {best_model_name} (Optimal)', fontsize=12, fontweight='bold')
        plt.ylabel('Actual Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        plt.savefig(os.path.join(self.static_img_dir, "confusion_matrix.png"), dpi=150)
        plt.close()

        # 5. Learning Curve (Best Model)
        train_sizes, train_scores, test_scores = learning_curve(
            best_model, X_train, y_train, cv=3, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
        )
        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="r", label="Training score")
        plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color="g", label="Cross-validation score")
        plt.xlabel("Training examples")
        plt.ylabel("Accuracy")
        plt.title(f"Learning Curves ({best_model_name})", fontsize=12, fontweight='bold')
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(self.static_img_dir, "learning_curve.png"), dpi=150)
        plt.close()

        # 6. PCA 2D Scatter Projection
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_test)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, cmap='coolwarm', alpha=0.6, edgecolors='none')
        plt.legend(handles=scatter.legend_elements()[0], labels=['No Flood', 'Flood'])
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA Feature Space projection (2D)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.static_img_dir, "pca_visualization.png"), dpi=150)
        plt.close()
