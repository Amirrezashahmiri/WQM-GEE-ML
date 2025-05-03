# GEE_Toolkit
A Google Earth Engine Toolkit and Machine Learning Framework for Monitoring Reservoir Water Quality Using Satellite Imagery and In Situ Data 

This repository contains:

A no-code Google Earth Engine (GEE) toolkit for computing and visualizing water-quality indices.

A Python machine learning pipeline for local calibration and prediction of chlorophyll‑a, turbidity, and water surface temperature (WST).

The toolkit automates:

Atmospheric correction and cloud masking of Landsat 8/9 & Sentinel‑2 imagery

Pixel-wise time-series extraction and regional clipping

Export of GeoTIFF composites to Google Drive

The ML pipeline provides:

Point-based extraction of in situ CTD measurements aligned with satellite pixels

Feature selection (RFE, SelectKBest) and hyperparameter tuning via RandomizedSearchCV

Ensemble modeling (Random Forest, XGBoost, CatBoost, MLP, stacking, voting)

Highlights

No-code GEE toolkit integrates satellite and in situ data for water quality models.

Automates atmospheric correction, cloud masking, and time-series extraction.

Compute empirical indices (NDVI, NDWI, LST, TSM, SPM, Chl‑a, turbidity, CDOM) in GEE.

Point-based extraction aligns CTD data with Landsat 8/9 & Sentinel-2 pixels.

ML models achieved R² of 0.70–0.84 for chlorophyll‑a, turbidity, and WST.
