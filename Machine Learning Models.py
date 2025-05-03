import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import os
from scipy.stats import randint, uniform
import seaborn as sns

# Set seaborn style for better aesthetics
sns.set(style='whitegrid')

# Set random seeds for reproducibility
np.random.seed(42)

# Define file paths (update these as per your local setup)
input_file = r"C:\Users\DFMRendering\Desktop\Microplastic\GEE_Dhaygah_Dam_300m\Calibration\Chl_a_Landsat\Data analysis and FS\selected_features.csv"
output_dir = r"C:\Users\DFMRendering\Desktop\Microplastic\GEE_Dhaygah_Dam_300m\Calibration\Chl_a_Landsat\Modeling"

# Create output directory if it doesn’t exist
os.makedirs(output_dir, exist_ok=True)

# Load and preprocess data
data = pd.read_csv(input_file)
target_column = "Chl-a [mg/l]"
exclude_columns = ['Lat', 'Long', 'Date', 'latitude', 'longitude', 'product', 'insitu_date']
feature_columns = [col for col in data.select_dtypes(include=[np.number]).columns if col != target_column and col not in exclude_columns]
data = data[feature_columns + [target_column]].dropna()

# Prepare features and target
X = data[feature_columns]
y = data[target_column]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.39, random_state=42)

# Report the number of training and testing samples
num_train = len(y_train)
num_test = len(y_test)

# Define selected models with descriptions
base_models = {
    'Linear Regression': LinearRegression(),  # Simple linear model assuming linear relationships
    'Random Forest': RandomForestRegressor(random_state=42),  # Tree-based ensemble for non-linear patterns
    'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42),  # Boosting model for high performance
    'CatBoost': CatBoostRegressor(random_state=42, verbose=0),  # Boosting optimized for categorical data
    'MLP': MLPRegressor(max_iter=1000, random_state=42),  # Neural network for complex patterns
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),  # Boosting ensemble for robust predictions
    'Stacking Regressor': StackingRegressor(
        estimators=[('rf', RandomForestRegressor(random_state=42)), ('xgb', xgb.XGBRegressor(random_state=42))],
        final_estimator=RidgeCV()
    ),  # Combines RF and XGBoost with RidgeCV as meta-learner
    'Stacking Regressor 2': StackingRegressor(
        estimators=[('mlp', MLPRegressor(max_iter=1000, random_state=42)), ('cat', CatBoostRegressor(random_state=42, verbose=0))],
        final_estimator=LinearRegression()
    ),  # Combines MLP and CatBoost with LinearRegression as meta-learner
    'Voting Regressor': VotingRegressor(
        estimators=[('rf', RandomForestRegressor(random_state=42)), ('xgb', xgb.XGBRegressor(random_state=42)), ('mlp', MLPRegressor(max_iter=1000, random_state=42))]
    )  # Averages predictions from RF, XGBoost, and MLP
}

# Model descriptions for reporting
model_descriptions = {
    'Linear Regression': 'A simple linear model that assumes a linear relationship between features and target.',
    'Random Forest': 'An ensemble of decision trees that captures non-linear relationships and reduces overfitting.',
    'XGBoost': 'A gradient boosting model optimized for speed and performance, effective for structured data.',
    'CatBoost': 'A boosting model designed to handle categorical features efficiently with minimal preprocessing.',
    'MLP': 'A multi-layer perceptron neural network capable of learning complex patterns in data.',
    'Gradient Boosting': 'A boosting ensemble that builds trees sequentially to minimize errors.',
    'Stacking Regressor': 'Combines Random Forest and XGBoost predictions using RidgeCV as a meta-learner.',
    'Stacking Regressor 2': 'Combines MLP and CatBoost predictions using LinearRegression as a meta-learner.',
    'Voting Regressor': 'Averages predictions from Random Forest, XGBoost, and MLP for a balanced output.'
}

# Define parameter distributions for tunable models
tunable_models = ['Random Forest', 'XGBoost', 'CatBoost', 'MLP', 'Gradient Boosting']

param_dist = {
    'Random Forest': {
        'n_estimators': randint(50, 200),
        'max_depth': [None] + list(range(5, 20)),
        'min_samples_split': randint(2, 11)
    },
    'XGBoost': {
        'n_estimators': randint(50, 200),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10)
    },
    'CatBoost': {
        'iterations': randint(50, 200),
        'learning_rate': uniform(0.01, 0.3),
        'depth': randint(3, 10)
    },
    'MLP': {
        'hidden_layer_sizes': [(50,), (100, 50), (100,)],
        'learning_rate_init': [0.001, 0.01],
        'alpha': [0.0001, 0.001]
    },
    'Gradient Boosting': {
        'n_estimators': randint(50, 200),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10)
    }
}

# Function to compute MAPE, handling zero values
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else None

# Initialize dictionary to store predictions and metrics
model_data = {}

# Train, tune, and evaluate each model
for model_name in base_models:
    print(f"\nTraining and evaluating {model_name}...")
    
    if model_name in tunable_models:
        # Perform RandomizedSearchCV
        randomized_search = RandomizedSearchCV(
            base_models[model_name],
            param_dist[model_name],
            n_iter=20,
            cv=5,
            scoring='neg_mean_squared_error',
            random_state=42,
            n_jobs=-1
        )
        randomized_search.fit(X_train, y_train)
        model = randomized_search.best_estimator_
        best_params = randomized_search.best_params_
    else:
        # For non-tunable models, train directly
        model = base_models[model_name]
        model.fit(X_train, y_train)
        best_params = None  # No parameters tuned for these models
    
    # Predict on both training and test sets
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Compute metrics for test set
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    mape = mean_absolute_percentage_error(y_test, y_pred_test)
    explained_variance = explained_variance_score(y_test, y_pred_test)
    
    # Store predictions and metrics
    model_data[model_name] = {
        'y_train': y_train,
        'y_pred_train': y_pred_train,
        'y_test': y_test,
        'y_pred_test': y_pred_test,
        'mse': mse,
        'r2': r2,
        'best_params': best_params
    }
    
    # Print results
    print(f"{model_name} MSE: {mse:.4f}")
    print(f"{model_name} R2: {r2:.4f}")
    print(f"{model_name} MAE: {mae:.4f}")
    print(f"{model_name} MAPE: {mape:.4f}%" if mape is not None else f"{model_name} MAPE: N/A")
    print(f"{model_name} Explained Variance: {explained_variance:.4f}")
    
    # Save detailed metrics and model information
    with open(os.path.join(output_dir, f"evaluation_metrics_{model_name.replace(' ', '_')}.txt"), "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Description: {model_descriptions[model_name]}\n")
        f.write(f"Number of training samples: {num_train}\n")
        f.write(f"Number of testing samples: {num_test}\n")
        if best_params:
            f.write("Best parameter combination from tuning:\n")
            for param, value in best_params.items():
                f.write(f"  {param}: {value}\n")
        else:
            f.write("No parameters were tuned for this model.\n")
        f.write("\nEvaluation Metrics on Test Set:\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"R2: {r2:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"MAPE: {mape:.4f}%\n" if mape is not None else "MAPE: N/A\n")
        f.write(f"Explained Variance: {explained_variance:.4f}\n")

# Create a grid of subplots for all models
num_models = len(base_models)
cols = 3  # Number of columns in the grid
rows = int(np.ceil(num_models / cols))  # Calculate rows needed

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True, sharey=True)
axes = axes.flatten()  # Flatten to 1D array for easy iteration

for idx, model_name in enumerate(base_models):
    ax = axes[idx]
    data = model_data[model_name]
    
    # Scatter plots for training and testing data with thin black boundaries
    ax.scatter(data['y_train'], data['y_pred_train'], alpha=0.5, color='blue', edgecolor='k', linewidth=0.5, label='Training Data', s=30)
    ax.scatter(data['y_test'], data['y_pred_test'], alpha=0.8, color='red', edgecolor='k', linewidth=0.6, label='Testing Data', s=45)
    
    # Ideal fit line with improved visibility
    ax.plot([min(y), max(y)], [min(y), max(y)], 'k--', label='Ideal Fit', linewidth=1.5)
    
    # Set title and labels with larger font sizes for professional look
    ax.set_title(model_name, fontsize=16, fontweight='bold')
    ax.set_xlabel('Actual Chl-a [mg/l]', fontsize=14)
    ax.set_ylabel('Predicted Chl-a [mg/l]', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add R² and MSE in the bottom-right corner with larger font size and bold
    metrics_text = f"R²: {data['r2']:.2f}\nMSE: {data['mse']:.2f}"
    ax.text(0.95, 0.05, metrics_text, transform=ax.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.5))
    
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, color='lightgray', alpha=0.3, linewidth=0.5)

# Hide any unused subplots
for idx in range(num_models, len(axes)):
    axes[idx].axis('off')

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "actual_vs_predicted_all_models.png"), dpi=400)
plt.close()

print("All models evaluated. Combined plot saved in the output directory.")