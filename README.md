````markdown
# GEE Toolkit for Reservoir Water Quality Monitoring

An open-source, no-code Google Earth Engine toolkit paired with a Python ML pipeline for end-to-end reservoir water quality assessment.

## Features

- **No-code GEE interface**: Atmospheric correction, cloud masking, and pixel‑wise time-series extraction.
- **Empirical indices**: Compute NDVI, NDWI, LST, TSM, SPM, NDSSI, TSS, Chl‑a, turbidity, CDOM, and cyanobacteria without coding.
- **Point-based extraction**: Align in situ CTD measurements with Landsat 8/9 & Sentinel‑2 pixels for local calibration.
- **Python ML pipeline**: Feature selection (RFE, SelectKBest), RandomizedSearchCV tuning, and ensemble models (RF, XGBoost, CatBoost, MLP, stacking, voting).
- **Proven performance**: Achieves R² of 0.70–0.84 for chlorophyll‑a, turbidity, and water surface temperature (WST).

## Getting Started

### Prerequisites

- Python 3.7+
- [Google Earth Engine Python API](https://developers.google.com/earth-engine/python_install)
- Git

### Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/GEE_Toolkit.git
cd GEE_Toolkit

# Setup Python environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r ml_pipeline/requirements.txt
````

## Usage

### 1. Launch the GEE Toolkit UI

```bash
cd gee_toolkit
python3 -m http.server 8000
```

Open your browser at `http://localhost:8000`, select your region, date range, and indices, then export composites to Google Drive.

### 2. Run the Python ML Pipeline

```bash
cd ml_pipeline
python run_pipeline.py \
  --ctd ../data/ctd_samples.csv \
  --satellite ../data/satellite_features.csv \
  --output ../results/model_performance.csv
```

Outputs include model metrics (R², MAE, MSE, MAPE) and feature importance summaries.

## Repository Structure

```
GEE_Toolkit/
├── gee_toolkit/          # No-code GEE UI and scripts
├── ml_pipeline/          # Python scripts for data merging & ML
├── data/                 # Sample CTD and satellite export files
├── results/              # Model performance outputs
├── notebooks/            # Jupyter notebooks for demos
└── README.md             # Project overview and instructions
```

## Contributing

Feel free to open issues or submit pull requests for enhancements and bug fixes.

## License

This project is licensed under the MIT License.

---

*Developed by Amirreza Shahmiri*

```
```
