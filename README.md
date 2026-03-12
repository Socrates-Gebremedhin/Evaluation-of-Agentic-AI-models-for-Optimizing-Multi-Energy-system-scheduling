## Project Overview

This repository contains the code and data used to evaluate agentic AI models for optimizing multi-energy system scheduling, including forecasting, optimization, and scheduling components.

## Directory Structure

- **Data/**: Input and processed datasets, including forecasts, feasibility statistics, and other CSV files used across the project.
- **energy_components/**: Models of the physical system components, such as the battery, boiler, CHP, PV, buffers, and the digital twin of the full system.
- **forecast_models/**: Implementations of the different forecasting models (LSTM, XGBoost, SARIMA/X, persistence, etc.) used for electricity, heat, and PV.
- **forecast_implementation/**: Notebooks and helper code that train, compare, and evaluate the forecast models.
- **preprocessor/**: Utilities for feature engineering and time-series preprocessing (lags, calendar features, weather data, reshaping to sequences, etc.).
- **Optimization/**: Feasibility layers and optimization utilities that check and enforce constraints for LP and RL-based scheduling.
- **Schedulers/**: Baseline controllers and look-ahead schedulers, plus runner scripts and notebooks used to execute and benchmark different strategies.
- **helper.py**: Small shared helper functions used by multiple modules.
- **split_config.csv**: Configuration file describing how the time series is split into training, validation, and test sets.
- **Data Analysis.ipynb**: Notebook with high-level exploratory data analysis and summarised results.
- **Evaluation of Agentic AI models for Optimizing Multi-Energy system scheduling.pdf**: The main report/thesis document that explains the methodology and results.

## Key Components

- **Data**: CSV files containing forecasts, feasibility statistics, and other inputs used by forecasting and optimization modules.
- **Preprocessing & Forecasting**: `preprocessor/`, `forecast_models/`, and `forecast_implementation/` together handle data preparation, feature creation, and training/evaluation of forecast models for electricity, heat, and PV.
- **Optimization & Scheduling**: `Optimization/` and `Schedulers/` implement feasibility layers, baseline controllers, and look-ahead LP/RL-based schedulers.
- **Energy Components**: `energy_components/` defines the digital twin of the energy system and its individual components used inside the schedulers and optimizers.
