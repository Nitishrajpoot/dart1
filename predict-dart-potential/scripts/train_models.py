"""
Train and evaluate machine learning models for DART prediction.

This script:
1. Loads the processed feature dataset
2. Splits data into train/test sets
3. Trains multiple classification models
4. Evaluates and compares model performance
5. Saves the best-performing model
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# Create output directory
os.makedirs('data/models', exist_ok=True)

def load_and_prepare_data():
    """Load and prepare the dataset for training."""
    print("=" * 60)
    print("Loading Data")
    print("=" * 60)
    
    # Load main dataset
    df = pd.read_csv('data/processed/dart_dataset_with_descriptors.csv')
    
    # Load fingerprints
    morgan_df = pd.read_csv('data/processed/morgan_fingerprints.csv')
    maccs_df = pd.read_csv('data/processed/maccs_fingerprints.csv')
    
    # Merge all features
    df = pd.merge(df, morgan_df, on='casrn', how='left')
    df = pd.merge(df, maccs_df, on='casrn', how='left')
    
    print(f"Total samples: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    
    # Separate features and labels
    # Exclude metadata columns
    exclude_cols = ['casrn', 'chemical_name', 'smiles', 'dart_label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    y = df['dart_label'].values
    
    # Handle missing values (replace -1 AC50 with 0, indicating no activity)
    X = np.nan_to_num(X, nan=0.0)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution: Positive={sum(y)}, Negative={len(y)-sum(y)}")
    
    return X, y, df, feature_cols

def train_and_evaluate_models(X, y, feature_cols):
    """Train multiple models and evaluate performance."""
    print("\n" + "=" * 60)
    print("Model Training and Evaluation")
    print("=" * 60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, 'data/models/scaler.pkl')
    print("Scaler saved to data/models/scaler.pkl")
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        )
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n{'-' * 60}")
        print(f"Training {name}...")
        print(f"{'-' * 60}")
        
        # Train model
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        if name == 'Logistic Regression':
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='balanced_accuracy')
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='balanced_accuracy')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")
        print(f"CV Balanced Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # Select best model based on balanced accuracy
    best_model_name = max(results, key=lambda x: results[x]['balanced_accuracy'])
    best_model = results[best_model_name]['model']
    
    print("\n" + "=" * 60)
    print(f"Best Model: {best_model_name}")
    print("=" * 60)
    print(f"Balanced Accuracy: {results[best_model_name]['balanced_accuracy']:.4f}")
    print(f"F1-Score: {results[best_model_name]['f1_score']:.4f}")
    print(f"AUC-ROC: {results[best_model_name]['auc_roc']:.4f}")
    
    # Save best model
    model_path = 'data/models/dart_model.pkl'
    joblib.dump(best_model, model_path)
    print(f"\nBest model saved to {model_path}")
    
    # Save feature names
    feature_names_path = 'data/models/feature_names.pkl'
    joblib.dump(feature_cols, feature_names_path)
    print(f"Feature names saved to {feature_names_path}")
    
    # Save model metadata
    metadata = {
        'model_name': best_model_name,
        'balanced_accuracy': results[best_model_name]['balanced_accuracy'],
        'f1_score': results[best_model_name]['f1_score'],
        'auc_roc': results[best_model_name]['auc_roc'],
        'cv_mean': results[best_model_name]['cv_mean'],
        'cv_std': results[best_model_name]['cv_std'],
        'n_features': len(feature_cols),
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    metadata_df = pd.DataFrame([metadata])
    metadata_df.to_csv('data/models/model_metadata.csv', index=False)
    print(f"Model metadata saved to data/models/model_metadata.csv")
    
    return results, best_model_name, y_test

def plot_model_comparison(results, y_test):
    """Create visualization comparing model performance."""
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Balanced Accuracy Comparison
    ax = axes[0, 0]
    model_names = list(results.keys())
    balanced_accs = [results[m]['balanced_accuracy'] for m in model_names]
    ax.bar(model_names, balanced_accs, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('Model Comparison: Balanced Accuracy')
    ax.set_ylim([0, 1])
    for i, v in enumerate(balanced_accs):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # 2. F1-Score Comparison
    ax = axes[0, 1]
    f1_scores = [results[m]['f1_score'] for m in model_names]
    ax.bar(model_names, f1_scores, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax.set_ylabel('F1-Score')
    ax.set_title('Model Comparison: F1-Score')
    ax.set_ylim([0, 1])
    for i, v in enumerate(f1_scores):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # 3. ROC Curves
    ax = axes[1, 0]
    for name in model_names:
        y_pred_proba = results[name]['y_pred_proba']
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = results[name]['auc_roc']
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Confusion Matrix (Best Model)
    ax = axes[1, 1]
    best_model_name = max(results, key=lambda x: results[x]['balanced_accuracy'])
    y_pred = results[best_model_name]['y_pred']
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(f'Confusion Matrix: {best_model_name}')
    
    plt.tight_layout()
    plt.savefig('data/models/model_comparison.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to data/models/model_comparison.png")
    plt.close()

if __name__ == '__main__':
    try:
        # Load data
        X, y, df, feature_cols = load_and_prepare_data()
        
        # Train and evaluate models
        results, best_model_name, y_test = train_and_evaluate_models(X, y, feature_cols)
        
        # Create visualizations
        plot_model_comparison(results, y_test)
        
        print("\n" + "=" * 60)
        print("Model Training Complete!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run the data collection and feature generation scripts first:")
        print("1. python scripts/fetch_toxcast_data.py")
        print("2. python scripts/fetch_dart_labels.py")
        print("3. python scripts/generate_features.py")
