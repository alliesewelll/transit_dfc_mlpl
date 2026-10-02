# Public Transit Ridership Forecasting

An end-to-end machine learning and time-series forecasting project for
predicting public transit ridership using historical ridership, service
levels, seasonal patterns, and external factors.

## Project Goals

- Explore historical public transit ridership patterns
- Analyze trends and seasonality
- Engineer temporal and lag-based features
- Incorporate external factors such as weather and economic conditions
- Compare baseline, machine learning, and time-series forecasting methods
- Evaluate models using chronological validation
- Deploy the selected model through a FastAPI endpoint

## Data

Primary ridership data is sourced from the Federal Transit Administration's
National Transit Database (NTD) Monthly Ridership dataset.

Additional external datasets will be incorporated during later stages of
the project.

## Current Dataset and Scope

This project forecasts monthly ridership for King County's directly
operated motorbus service in the Seattle area.

- **Source:** FTA National Transit Database, July 2026 Complete Monthly
  Ridership release, including adjustments and estimates
- **Agency:** King County (`00001`)
- **Mode:** Motorbus (`MB`)
- **Type of service:** Directly operated (`DO`)
- **Target:** Unlinked passenger trips (boardings, not unique passengers)
- **Coverage:** January 2002–July 2026, with 295 monthly observations

This series excludes separately reported trolleybus and contracted
motorbus services.

## Data Preparation

Run `notebooks/01_eda.ipynb` to select the series, reshape month columns
into rows, check data quality, and export the cleaned CSV.

The notebook checks for duplicate dates, chronological ordering,
missing months, missing ridership values, and negative ridership.

The original Excel workbook remains unchanged. Historical values may
include FTA adjustments, estimates, and revisions.