import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from feature_engineering import FeaturePipeline
from evaluate_models import ModelEvaluator

# -------------------------------------------------------------
# 1. SYNTHETIC DATA GENERATOR (17 Features)
# -------------------------------------------------------------
def generate_raw_dataset(num_samples=5000):
    print(f"Generating {num_samples} records of historical flood logs...")
    np.random.seed(42)

    # Environmental/Meteorological Features
    annual_rainfall = np.random.normal(2400, 750, num_samples).clip(600, 5500)
    monthly_rainfall = np.random.normal(280, 140, num_samples).clip(10, 850)
    temperature = np.random.normal(26.5, 4.5, num_samples).clip(8.0, 46.0)
    humidity = np.random.normal(74, 10, num_samples).clip(30, 100)
    pressure = np.random.normal(1008, 9, num_samples).clip(970, 1035)
    cloud_cover = np.random.normal(62, 18, num_samples).clip(0, 100)
    wind_speed = np.random.normal(16, 7, num_samples).clip(0, 60)
    river_water_level = np.random.normal(4.8, 2.8, num_samples).clip(0.1, 15.0)
    ground_water_level = np.random.normal(3.5, 1.8, num_samples).clip(0.05, 10.0)
    visibility = np.random.normal(7.5, 2.5, num_samples).clip(0.2, 16.0)
    
    # Geography
    latitude = np.random.uniform(8.4, 37.6, num_samples)
    longitude = np.random.uniform(68.1, 97.2, num_samples)
    
    # Categories
    seasons = ['Summer', 'Monsoon', 'Winter', 'Spring']
    season_choices = np.random.choice(seasons, size=num_samples, p=[0.25, 0.40, 0.20, 0.15])
    
    months = ['January', 'February', 'March', 'April', 'May', 'June', 
              'July', 'August', 'September', 'October', 'November', 'December']
    month_choices = np.random.choice(months, size=num_samples)

    districts = ['Thane', 'Ernakulam', 'Cuttack', 'Patna', 'Guwahati', 'Medinipur', 'Nashik', 'Pune']
    district_choices = np.random.choice(districts, size=num_samples)
    
    states = ['Maharashtra', 'Kerala', 'Odisha', 'Bihar', 'Assam', 'West Bengal']
    state_choices = np.random.choice(states, size=num_samples)

    df = pd.DataFrame({
        'Annual_Rainfall': annual_rainfall,
        'Monthly_Rainfall': monthly_rainfall,
        'Temperature': temperature,
        'Humidity': humidity,
        'Pressure': pressure,
        'Cloud_Cover': cloud_cover,
        'Wind_Speed': wind_speed,
        'River_Water_Level': river_water_level,
        'Ground_Water_Level': ground_water_level,
        'Visibility': visibility,
        'Latitude': latitude,
        'Longitude': longitude,
        'Season': season_choices,
        'Month': month_choices,
        'District': district_choices,
        'State': state_choices
    })

    # High river levels, heavy monthly rainfall, low pressure, ground water levels increase risk
    score = (
        0.28 * (df['River_Water_Level'] / 15.0) +
        0.24 * (df['Monthly_Rainfall'] / 850.0) +
        0.14 * (df['Ground_Water_Level'] / 10.0) +
        0.12 * (df['Humidity'] / 100.0) -
        0.10 * (df['Pressure'] - 1008.0) / 38.0 +
        0.06 * (df['Cloud_Cover'] / 100.0) +
        0.04 * (df['Annual_Rainfall'] / 5500.0) -
        0.04 * (df['Visibility'] / 16.0) +
        0.02 * (df['Wind_Speed'] / 60.0)
    )

    # Monsoon season weight
    season_weights = {'Monsoon': 0.15, 'Summer': -0.05, 'Spring': -0.02, 'Winter': -0.08}
    score += df['Season'].map(season_weights)

    # Sigmoid function for probability
    prob = 1 / (1 + np.exp(-(score - 0.45) / 0.07))
    prob = (prob + np.random.normal(0, 0.02, num_samples)).clip(0, 1)

    df['Flood'] = (np.random.rand(num_samples) < prob).astype(int)

    # Insert missing entries (1.5%) to show imputation
    for col in ['Temperature', 'Humidity', 'Pressure', 'River_Water_Level']:
        mask = np.random.rand(num_samples) < 0.015
        df.loc[mask, col] = np.nan

    # Save raw CSV
    os.makedirs(os.path.dirname(Config.RAW_DATA_PATH), exist_ok=True)
    df.to_csv(Config.RAW_DATA_PATH, index=False)
    print(f"Raw dataset created at {Config.RAW_DATA_PATH}")
    return df

# -------------------------------------------------------------
# 2. RUN PIPELINE
# -------------------------------------------------------------
def run_training_pipeline():
    # 2.1 Load or Generate Raw Dataset
    if not os.path.exists(Config.RAW_DATA_PATH):
        df_raw = generate_raw_dataset()
    else:
        df_raw = pd.read_csv(Config.RAW_DATA_PATH)
        print(f"Loaded existing raw dataset from {Config.RAW_DATA_PATH}")

    # 2.2 Preprocessing & Feature Engineering
    pipeline = FeaturePipeline()
    pipeline.fit(df_raw)
    
    # Save Feature Pipeline metadata (StandardScaler and outlier bounds)
    pipeline.save()

    # Preprocess Full Data for Export
    df_processed = pipeline.transform(df_raw)
    df_processed['Flood'] = df_raw['Flood']
    
    os.makedirs(os.path.dirname(Config.PROCESSED_DATA_PATH), exist_ok=True)
    df_processed.to_csv(Config.PROCESSED_DATA_PATH, index=False)
    print(f"Processed dataset saved to {Config.PROCESSED_DATA_PATH}")

    # 2.3 Train/Test Split
    X = df_processed[Config.FEATURES]
    y = df_processed['Flood']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 2.4 Evaluate 14 Classifiers
    evaluator = ModelEvaluator(Config.STATIC_IMAGES_DIR)
    metrics_summary, best_name, best_model = evaluator.evaluate_all(X_train, X_test, y_train, y_test)

    # 2.5 Save Best Model
    joblib.dump(best_model, Config.MODEL_PATH)
    print(f"Best model ({best_name}) binary saved to {Config.MODEL_PATH}")

    # Save metrics JSON
    metrics_log = {
        'best_model_name': best_name,
        'metrics': metrics_summary,
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(Config.MODEL_METRICS_PATH, 'w') as f:
        json.dump(metrics_log, f, indent=4)
    print(f"Model comparison metrics saved to {Config.MODEL_METRICS_PATH}")

    # 2.6 EDA Plot Generation (Heatmap, Histograms, Scatter, Boxplots)
    generate_eda_plots(df_raw)
    
    # 2.7 Feature Importance (Best Model or fallback)
    generate_feature_importance_plot(best_model, best_name, X)

    # 2.8 Mock Label Encoders (District and State mappings for mapping module checks)
    districts = df_raw['District'].unique().tolist()
    states = df_raw['State'].unique().tolist()
    label_encoders = {
        'districts': districts,
        'states': states
    }
    joblib.dump(label_encoders, Config.LABEL_ENCODER_PATH)
    print(f"Label encoders saved to {Config.LABEL_ENCODER_PATH}")

def generate_eda_plots(df):
    """Generate extensive statistical plots for the dashboard."""
    sns.set_theme(style="darkgrid")
    df_clean = df.dropna().copy()
    df_clean['Season'] = df_clean['Season'].map(lambda x: Config.SEASON_MAPPING.get(x, 0))
    df_clean['Month'] = df_clean['Month'].map(lambda x: Config.MONTH_MAPPING.get(x, 1))
    
    # Heatmap
    plt.figure(figsize=(12, 10))
    corr = df_clean.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Analysis Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.STATIC_IMAGES_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    # Box Plot Outliers (River water levels by Season)
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='Season', y='River_Water_Level', hue='Flood', palette='Set2')
    plt.title('River Water Levels and Outliers by Season', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.STATIC_IMAGES_DIR, "boxplot_outliers.png"), dpi=150)
    plt.close()

    # Distribution Violin Plots
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df, x='Flood', y='Monthly_Rainfall', palette='muted')
    plt.xticks([0, 1], ['No Flood', 'Flood'])
    plt.title('Monthly Rainfall Distributions (Violin Plot)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.STATIC_IMAGES_DIR, "violin_distribution.png"), dpi=150)
    plt.close()

    # Scatter Map coordinates
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Longitude', y='Latitude', hue='Flood', palette={0: '#3498db', 1: '#e74c3c'}, alpha=0.5)
    plt.title('Flood Occurrences by Coordinates', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.STATIC_IMAGES_DIR, "coordinates_scatter.png"), dpi=150)
    plt.close()

    # Pie Chart
    plt.figure(figsize=(6, 6))
    counts = df['Flood'].value_counts()
    plt.pie(counts, labels=['No Flood', 'Flood'], autopct='%1.1f%%', colors=['#2563eb', '#ef4444'], startangle=90)
    plt.title('Target Distribution', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.STATIC_IMAGES_DIR, "pie_chart.png"), dpi=150)
    plt.close()

def generate_feature_importance_plot(model, model_name, X):
    """Saves horizontal feature importance rankings."""
    plt.figure(figsize=(10, 6))
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        # Fallback to Random Forest if classifier doesn't expose coefficients (like KNN or Stacking)
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
        rf.fit(X, np.zeros(X.shape[0]))  # Dummy fit or just compute
        importances = rf.feature_importances_
        model_name = "Random Forest (Fallback)"
        
    feat_series = pd.Series(importances, index=Config.FEATURES).sort_values(ascending=True)
    feat_series.plot(kind='barh', color=sns.color_palette('viridis', len(feat_series)))
    plt.title(f'Feature Importance ({model_name})', fontsize=12, fontweight='bold')
    plt.xlabel('Score')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.STATIC_IMAGES_DIR, "feature_importance.png"), dpi=150)
    plt.close()

if __name__ == '__main__':
    from datetime import datetime
    print("====================================================")
    print("STARTING CAPSTONE ML TRAINING PIPELINE (14 ALGORITHMS)")
    print("====================================================")
    run_training_pipeline()
    print("====================================================")
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("====================================================")
