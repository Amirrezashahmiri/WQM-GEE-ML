import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split

# Define the path to the input file (adjust as needed)
input_file = r"C:\Users\DFMRendering\Desktop\Microplastic\GEE_Dhaygah_Dam_300m\Calibration\Chl_a_Landsat\landsat_matched_no_outliers.csv"

# Define the output directory (adjust as needed)
output_dir = r"C:\Users\DFMRendering\Desktop\Microplastic\GEE_Dhaygah_Dam_300m\Calibration\Chl_a_Landsat\Data analysis and FS"

# Set the font to Times New Roman for all plots and improve readability
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20  # Base font size

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the data from the CSV file
df = pd.read_csv(input_file)

# Replace underscores with spaces in all column names
df.columns = [col.replace('_', ' ') for col in df.columns]

# Define the target variable
target_column = "Chl-a [mg/l]"

# Define the columns to exclude from features (with spaces to match renamed columns)
exclude_columns = ["Lat", "Long", "latitude", "longitude", "Date", "insitu date", "product date", "time diff", target_column]

# Select feature columns (all columns except excluded ones)
feature_columns = [col for col in df.columns if col not in exclude_columns]

# Select only numeric feature columns
numeric_feature_columns = df[feature_columns].select_dtypes(include=[np.number]).columns

# Separate features and target
features = df[numeric_feature_columns]
target = df[target_column]

# Combine features and target to drop rows with any missing values
combined = pd.concat([features, target], axis=1)
combined = combined.dropna()

# Separate features and target again after dropping NaNs
features = combined.drop(target_column, axis=1)
target = combined[target_column]

# Define units for specific bands
unit_dict = {
    "SPM landsa Qiu": "g/m³",
    "LST Landsat": "°C",
    # Add other specific units if needed
}

# Calculate correlation matrix including the target
corr_matrix = combined.corr()

# Plot correlation matrix heatmap with 'RdBu_r' colormap
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, cmap="RdBu_r", fmt=".2f", linewidths=0.5, annot_kws={"size": 10}, cbar_kws={'shrink': 0.8})
plt.title("Correlation Matrix Heatmap (Including Target)", fontsize=20, pad=20)
plt.xticks(rotation=45, ha="right", fontsize=18)
plt.yticks(rotation=0, fontsize=18)
plt.savefig(os.path.join(output_dir, "correlation_heatmap_with_target.png"), dpi=400, bbox_inches="tight")
plt.close()

# Step 1: Identify and drop highly correlated features (correlation > 0.9)
high_corr_threshold = 0.9
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > high_corr_threshold and corr_matrix.columns[i] != target_column and corr_matrix.columns[j] != target_column:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

# Drop one feature from each highly correlated pair
features_to_drop = set()
for pair in high_corr_pairs:
    if pair[0] not in features_to_drop and pair[1] not in features_to_drop:
        corr_with_target0 = abs(corr_matrix.loc[pair[0], target_column])
        corr_with_target1 = abs(corr_matrix.loc[pair[1], target_column])
        if corr_with_target0 > corr_with_target1:
            features_to_drop.add(pair[1])
        else:
            features_to_drop.add(pair[0])

# Remove highly correlated features
features = features.drop(columns=features_to_drop)

# Standardize the remaining features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
y = target.values

# Step 2: Apply Recursive Feature Elimination (RFE)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rfe = RFE(estimator=rf, n_features_to_select=5, step=1)
rfe.fit(X_scaled, y)
rfe_selected_features = features.columns[rfe.support_].tolist()

# Step 3: Apply SelectKBest using mutual information
kbest = SelectKBest(score_func=mutual_info_regression, k=5)
kbest.fit(X_scaled, y)
kbest_selected_features = features.columns[kbest.get_support()].tolist()

# Step 4: Find intersection of selected features from RFE and SelectKBest
selected_features = list(set(rfe_selected_features).intersection(set(kbest_selected_features)))

# If intersection is empty, fallback to union
if not selected_features:
    selected_features = list(set(rfe_selected_features).union(set(kbest_selected_features)))
    print("No common features selected. Using union of RFE and SelectKBest features.")

# Save the selected features to a CSV file
selected_df = df[selected_features + [target_column]]
selected_df.to_csv(os.path.join(output_dir, "selected_features.csv"), index=False)

# Define a consistent color palette with 'RdBu_r'
colors = sns.color_palette("RdBu_r", n_colors=len(selected_features) + 1)

# Plot Raincloud Plots for selected features and target with units
def plot_raincloud(data, ax, label, color):
    unit = unit_dict.get(label, "Unknown")
    title = f"{label} [{unit}]" if unit != "Unknown" else label
    sns.violinplot(x=data, ax=ax, orient='h', inner=None, color=color, alpha=0.3)
    sns.boxplot(x=data, ax=ax, orient='h', width=0.1, color='black')
    sns.stripplot(x=data, ax=ax, orient='h', jitter=True, color='black', alpha=0.6)
    ax.set_title(title, fontsize=22, pad=10)
    ax.set_yticks([])
    ax.set_xlabel("")

n_vars = len(selected_features) + 1
n_cols = 3
n_rows = (n_vars + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows), sharey=False)
axes = axes.flatten()

for i, var in enumerate(selected_features + [target_column]):
    ax = axes[i]
    color = colors[i]
    plot_raincloud(selected_df[var], ax, var, color)

# Remove empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(pad=3.0)
plt.savefig(os.path.join(output_dir, "raincloud_plots.png"), dpi=400, bbox_inches="tight")
plt.close()

# Generate a report including units from unit_dict
with open(os.path.join(output_dir, "feature_selection_report.txt"), "w") as f:
    f.write("Feature Selection Report\n")
    f.write("========================\n\n")
    f.write("Features dropped due to high correlation (>0.9):\n")
    f.write(", ".join(features_to_drop) + "\n\n")
    f.write("Features selected by RFE:\n")
    f.write(", ".join(rfe_selected_features) + "\n\n")
    f.write("Features selected by SelectKBest (mutual information):\n")
    f.write(", ".join(kbest_selected_features) + "\n\n")
    f.write("Final selected features (intersection or union):\n")
    f.write(", ".join(selected_features) + "\n\n")
    f.write("Parameter Units:\n")
    for col in selected_features + [target_column]:
        unit = unit_dict.get(col, "Unknown")
        f.write(f"{col}: {unit}\n")
    f.write("\nSelected features saved to 'selected_features.csv'\n")

# Generate README text file
with open(os.path.join(output_dir, "README.txt"), "w") as f:
    f.write("README: Feature Selection Script\n")
    f.write("================================\n\n")
    f.write("This script performs feature selection on a dataset for predicting Chlorophyll-a (Chl-a) levels using Landsat data.\n")
    f.write("Below is a step-by-step explanation of what the code does:\n\n")
    f.write("1. **Define Paths and Directories:**\n")
    f.write("   - The script defines the input file path and output directory. Users must replace these paths with their own.\n")
    f.write("2. **Load Data:**\n")
    f.write("   - The CSV file specified in 'input_file' is loaded into a pandas DataFrame.\n")
    f.write("3. **Define Target and Exclude Columns:**\n")
    f.write("   - The target variable is set to 'Chl-a [mg/l]'.\n")
    f.write("   - Columns such as latitude, longitude, and dates are excluded from the feature set.\n")
    f.write("4. **Select Feature Columns:**\n")
    f.write("   - Features are selected by excluding non-numeric and predefined columns.\n")
    f.write("5. **Handle Missing Values:**\n")
    f.write("   - Rows with missing values in features or target are removed.\n")
    f.write("6. **Calculate Correlation Matrix:**\n")
    f.write("   - A correlation matrix is computed and saved as a heatmap image with 'RdBu_r' colormap.\n")
    f.write("7. **Identify and Drop Highly Correlated Features:**\n")
    f.write("   - Features with a correlation greater than 0.9 are identified, and one from each pair is dropped.\n")
    f.write("8. **Standardize Features:**\n")
    f.write("   - The remaining features are standardized using StandardScaler.\n")
    f.write("9. **Apply Recursive Feature Elimination (RFE):**\n")
    f.write("   - RFE with a RandomForestRegressor selects the top 6 features.\n")
    f.write("10. **Apply SelectKBest:**\n")
    f.write("    - SelectKBest with mutual information selects the top 6 features.\n")
    f.write("11. **Find Intersection or Union of Selected Features:**\n")
    f.write("    - The intersection of features from RFE and SelectKBest is used; if empty, the union is taken.\n")
    f.write("12. **Save Selected Features:**\n")
    f.write("    - The selected features and target are saved to 'selected_features.csv'.\n")
    f.write("13. **Generate Raincloud Plots:**\n")
    f.write("    - Raincloud plots are created with 'RdBu_r' palette and units for specific bands.\n")
    f.write("14. **Generate Report:**\n")
    f.write("    - A report is written detailing dropped features, selected features, and units.\n")
    f.write("15. **Print Completion Message:**\n")
    f.write("    - A message is printed to indicate the process is complete.\n\n")
    f.write("User Instructions:\n")
    f.write("- **Input File Path:** Replace 'input_file' with the path to your CSV file.\n")
    f.write("- **Output Directory:** Replace 'output_dir' with your desired output directory.\n")
    f.write("- Do not modify other parts unless adjusting the target or excluded columns.\n")

print("\nFeature selection process completed. Report and selected features saved.")
