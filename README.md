# A Google Earth Engine Toolkit for Reservoir Water Quality Monitoring

## Overview

This repository contains:

* A **no-code Google Earth Engine (GEE) toolkit** for computing and visualizing water-quality indices.
* A **Python machine learning pipeline** for local calibration and prediction of chlorophyll‑a, turbidity, and water surface temperature (WST).

> **Citation**  
> This work is currently under review in *Environmental Modelling & Software*.  
> If you use this toolkit or pipeline, please cite as follows:  
>
> Yavari, Z., Shahmiri, A. R., & Nikoo, M. R. (2025). *A Google Earth Engine toolkit for reservoir water quality monitoring*. Manuscript submitted for publication in *Environmental Modelling & Software*.

The toolkit automates:

* Atmospheric correction and cloud masking of Landsat 8/9 & Sentinel‑2 imagery
* Pixel-wise time-series extraction and regional clipping
* Export of GeoTIFF composites to Google Drive

The ML pipeline provides:

* Point-based extraction of in situ CTD measurements aligned with satellite pixels
* Feature selection (RFE, SelectKBest) and hyperparameter tuning via RandomizedSearchCV
* Ensemble modeling (Random Forest, XGBoost, CatBoost, MLP, stacking, voting)

## Highlights

* No-code GEE toolkit integrates satellite and in situ data for water quality models.
* Automates atmospheric correction, cloud masking, and time-series extraction.
* Compute empirical indices (NDVI, NDWI, LST, TSM, SPM, Chl‑a, turbidity, CDOM) in GEE.
* Point-based extraction aligns CTD data with Landsat 8/9 & Sentinel-2 pixels.
* ML models achieved R² of 0.70–0.84 for chlorophyll‑a, turbidity, and WST.


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/GEE_Toolkit.git
   cd GEE_Toolkit
   ```
2. Create a Python environment and install dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r ml_pipeline/requirements.txt
   ```
3. Configure GEE credentials by following the [GEE Python API setup guide](https://developers.google.com/earth-engine/python_install).

## Usage

### 1. Google Earth Engine Toolkit

* Open `gee_toolkit/index.html` in your web browser (hosted via local server or GitHub Pages).
* Select date range, indices, and region of interest.
* Export processed layers to Google Drive.

### 2. Python ML Pipeline

```bash
cd ml_pipeline
python run_pipeline.py \
  --ctd data/ctd_samples.csv \
  --satellite-exports data/satellite_features.csv \
  --output results/model_performance.csv
```

* **run\_pipeline.py** performs data merging, feature selection, hyperparameter tuning, and model evaluation.
* Results including R², MAE, MSE, and MAPE are saved in `results/`.

## Contributing

Contributions are welcome! Please open issues or submit pull requests with enhancements or bug fixes.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
