import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
import re
import shap

# ========== 1. Environment Settings ==========
warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
plt.rcParams['font.sans-serif'] = ['Arial'] 
plt.rcParams['axes.unicode_minus'] = False 

# ========== 2. File Paths ==========
curr_path = os.path.dirname(os.path.abspath(__file__))
base_csv_path = os.path.join(curr_path, 'Base_Data_Final.csv')
test_csv_path = os.path.join(curr_path, 'test_data_for_F.csv')

# Model priority list
model_files = ['lgbm_model.pkl', 'stack_model.pkl', 'xgb_model.pkl']

# ========== 3. Data Cleaning Pipeline ==========
def clean_numeric_strings(x):
    """Removes non-numeric characters such as brackets and quotes."""
    if isinstance(x, str):
        x = x.strip()
        # Keep only numbers, decimal points, signs, and scientific notation (E)
        x = re.sub(r"[^0-9eE\.\-\+]", "", x)
        return x
    return x

print("🚀 Starting integration task...")

# ========== 4. Data Loading and Preprocessing ==========
try:
    print("1/5 Loading and cleaning data...")
    base_df = pd.read_csv(base_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Apply cleaning to all columns in the test dataset
    for col in test_df.columns:
        test_df[col] = test_df[col].apply(clean_numeric_strings)
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
    
    # Fill missing values resulting from cleaning
    test_df = test_df.fillna(0) 

    # Identify target columns and separate features (X)
    target_candidates = ['price', 'log_price', 'log1p_price']
    drop_cols = [c for c in target_candidates if c in test_df.columns]
    X = test_df.drop(columns=drop_cols)
    
    print(f"✅ Data processed successfully. Feature count: {X.shape[1]}")
except Exception as e:
    print(f"❌ Data loading failed: {e}")
    exit()

# ========== 5. Model Loading (Backup Strategy) ==========
model = None
used_model_name = ""
for m_name in model_files:
    m_path = os.path.join(curr_path, m_name)
    if os.path.exists(m_path):
        try:
            with open(m_path, 'rb') as f:
                model = pickle.load(f)
            used_model_name = m_name
            print(f"2/5 Loaded model: {used_model_name}")
            break
        except:
            continue

if model is None:
    print("❌ No valid .pkl model file found in directory.")
    exit()

# ========== 6. Visualizations ==========

# 6.1 Spatial Distribution Map
print(f"3/5 Generating price spatial map...")
plt.figure(figsize=(12, 8))
plt.scatter(base_df['longitude'], base_df['latitude'], 
            c=np.log1p(base_df['price']), cmap='Spectral_r', s=10, alpha=0.6)
plt.colorbar(label='Log(Price)')
plt.title('Hong Kong Airbnb Price Spatial Distribution')
plt.savefig(os.path.join(curr_path, 'price_map_F.png'), dpi=300)
plt.close()

# 6.2 Feature Importance Plot
print(f"4/5 Generating feature importance plot...")
plt.figure(figsize=(10, 8))
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:] # Top 15 features
    plt.barh(range(len(indices)), importances[indices], color='teal')
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.title(f'Top 15 Features ({used_model_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(curr_path, 'feature_importance_F.png'), dpi=300)
plt.close()

# 6.3 SHAP Explanation
print(f"5/5 Calculating SHAP values using {used_model_name}...")
try:
    # Use 100 samples for performance
    X_sample = X.head(100).astype(float)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Handle SHAP output format for multi-output models
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Impact Analysis (Target: log1p price)")
    plt.tight_layout()
    plt.savefig(os.path.join(curr_path, 'shap_summary_F.png'), dpi=300)
    plt.close()
    print("✅ SHAP analysis complete.")
except Exception as e:
    print(f"⚠️ SHAP calculation error: {e}")

print(f"\n✨ Tasks completed successfully.")