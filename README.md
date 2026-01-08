# 8-K Item Timeliness and Materiality Dashboard

An interactive dashboard for analyzing regression results examining the timeliness (currentness) and materiality of 8-K disclosure items.

## Overview

This dashboard allows researchers to dynamically explore how different 8-K items are classified as current and/or material based on customizable parameters including time windows, significance thresholds, and materiality thresholds.

## Features

- **Dynamic Classification**: Items are automatically classified into four categories:
  - Current & Material
  - Current but Not Material
  - Material but Not Current
  - Neither Current nor Material

- **Interactive Controls**:
  - Time horizon selection (48h, 24h, 2h, 1h)
  - Adjustable significance threshold (1-20%)
  - Adjustable materiality threshold (5-50% of residual SD)
  - Option to scale coefficients by residual standard deviations

- **Retail Volume Flag**: Highlights items where retail investors respond even when total volume doesn't show significance

- **Visualizations**: Three horizontal bar charts (one per dependent variable) showing pre/post coefficients with 95% confidence intervals

- **Export Options**: Download classification table as CSV and charts as PNG files

## Required Files

The dashboard requires two CSV files in the same directory:

### 1. regression_results.csv
Contains regression coefficients with the following columns:
- `Item`: Item identifier (e.g., "Item 2.02")
- `Item_Description`: Full description of the item
- `Window`: Time window (e.g., "24h_before", "24h_after")
- `Dependent_Variable`: One of "Total_Volume", "Retail_Volume", "Absolute_Returns"
- `Coefficient`: Regression coefficient estimate
- `T_Statistic`: t-statistic for the coefficient

Should contain 648 rows (27 items × 8 windows × 3 dependent variables)

### 2. residual_sds.csv
Contains residual standard deviations with the following columns:
- `Dependent_Variable`: One of "Total_Volume", "Retail_Volume", "Absolute_Returns"
- `Window`: Time window (e.g., "24h_before", "24h_after")
- `Residual_SD`: Residual standard deviation value

Should contain 24 rows (3 dependent variables × 8 windows)

## Installation

### Local Installation

1. Clone or download this repository

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Place your data files (`regression_results.csv` and `residual_sds.csv`) in the same directory as `dashboard.py`

4. Run the dashboard:
```bash
streamlit run dashboard.py
```

5. The dashboard will open in your default web browser (typically at http://localhost:8501)

## Deployment to Streamlit Cloud

1. Create a GitHub repository and push these files:
   - `dashboard.py`
   - `requirements.txt`
   - `regression_results.csv`
   - `residual_sds.csv`

2. Go to [share.streamlit.io](https://share.streamlit.io)

3. Sign in with your GitHub account

4. Click "New app"

5. Select your repository, branch, and set `dashboard.py` as the main file

6. Click "Deploy"

Your dashboard will be publicly accessible within a few minutes!

## Usage Guide

### Controls (Left Sidebar)

1. **Time Horizon**: Select the time window for analysis (automatically selects symmetric before/after windows)

2. **Significance Level**: Set the threshold for determining statistical significance of "current" items

3. **Materiality Threshold**: Set the threshold (as % of residual SD) for determining material items

4. **Scale Coefficients**: Toggle to view coefficients scaled by their residual standard deviations

### Main Display

#### Summary Statistics
Shows count of items in each classification category under current settings

#### Classification Table
- Items organized into four sections by classification
- Shows all six coefficients (pre/post for each dependent variable)
- Includes significance stars (*** p<0.01, ** p<0.05, * p<0.10)
- Retail flag (🔔) indicates when retail responds but total volume doesn't
- Download as CSV button

#### Visualizations
- Three separate bar charts (Total Volume, Retail Volume, Absolute Returns)
- Items sorted by post-period coefficient (descending)
- Green bars = post-period coefficients
- Red bars = pre-period coefficients
- Error bars show 95% confidence intervals
- Individual download buttons for each chart

## Classification Logic

### Current
An item is "current" when:
- Post-period Total Volume coefficient is positive AND statistically significant
- Pre-period Total Volume coefficient is either insignificant OR negative

### Material
An item is "material" when:
- Sum of pre-period and post-period Absolute Returns coefficients exceeds the materiality threshold × residual SD
- (Note: This is algebraic sum, so negative values can offset positive values)

### Retail Flag
An item receives a retail flag when:
- Post-period Retail Volume is positive AND significant
- Post-period Total Volume is either insignificant OR not positive

## Technical Details

- **Framework**: Streamlit (easy deployment and natural dynamic updates)
- **Visualization**: Plotly (interactive charts with export capability)
- **Statistics**: SciPy (for critical value calculations)
- **Data Processing**: Pandas (efficient data manipulation)

## Sample Data

The repository includes sample data files with realistic values for testing. Replace these with your actual regression results before deploying.

## Troubleshooting

**Dashboard won't start**
- Ensure all packages are installed: `pip install -r requirements.txt`
- Check that data files are in the correct location

**Data not displaying**
- Verify CSV file column names match exactly (case-sensitive)
- Check that data files are not corrupted
- Ensure CSV files use proper formatting

**Charts not downloading**
- Install kaleido: `pip install kaleido`
- Kaleido is required for static image export

## Contact

For questions or issues, please contact the research team.

## License

This dashboard is provided for research purposes.
