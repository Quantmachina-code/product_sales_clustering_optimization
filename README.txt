# Product Segmentation Pipeline

Automated pipeline for classifying products into three business segments:
- **KVI (Key Value Items)**: Price-sensitive products that drive customer traffic
- **SD (Sales Drivers)**: High-volume products that maximize sales
- **PG (Profit Generators)**: High-margin products that maximize profitability

## Project Structure
```
project_root/
├── config/
│   └── config.yaml                    # Configuration (optional, not implemented)
│
├── data/
│   ├── raw/                           # Original dataset (data.parquet)
│   ├── processed/                     # Intermediate outputs
│   └── results/                       # Final predictions CSV
│
├── models/
│   ├── robust_scaler.pkl              # Fitted RobustScaler
│   ├── decision_tree.pkl              # Decision Tree classifier
│   ├── monotonic_clustering.pkl       # Hierarchical clustering + KNN
│   ├── gmm_model.pkl                  # Gaussian Mixture Model
│   └── feature_metadata.pkl           # Feature names and metadata
│
├── src/
│   ├── data/
│   │   ├── loader.py                  # Load CSV/Parquet data
│   │   ├── validator.py               # (not implemented)
│   │   └── preprocessor.py            # Violation detection, filtering, adjustment
│   │
│   ├── features/
│   │   ├── engineer.py                # Feature engineering (11 derived features)
│   │   ├── scaler.py                  # RobustScaler operations
│   │   └── outliers.py                # Multivariate outlier detection (Isolation Forest)
│   │
│   ├── models/
│   │   ├── decision_tree.py           # Decision Tree training
│   │   ├── clustering.py              # Hierarchical clustering training
│   │   └── gmm.py                     # GMM training
│   │
│   └── utils/
│       ├── io_utils.py                # Save/load model utilities
│
├── scripts/
│   ├── train.py                       # Training pipeline
│   ├── predict.py                     # Prediction pipeline
│
├── notebooks/                         # Exploration notebooks
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd project_root
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your data file:
```bash
# Ensure data.parquet is in data/raw/
data/raw/data.parquet
```

## Usage

### Training Pipeline

Train all models (Decision Tree, Hierarchical Clustering, GMM):
```bash
python scripts/train.py
```

**What happens during training:**
1. Loads raw data from `data/raw/data.parquet`
2. Preprocesses data (filters violations, removes duplicates, converts types)
3. Engineers 11 derived features
4. Detects and removes multivariate outliers (5% contamination)
5. Fits RobustScaler on clean data
6. Trains Decision Tree on labeled samples (54 samples)
7. Extracts top 4 features from Decision Tree
8. Trains Hierarchical Clustering on unlabeled data (top 4 features)
9. Trains GMM on unlabeled data (top 4 features)
10. Saves all models to `models/` directory

**Expected output:**
```
models/
├── robust_scaler.pkl
├── decision_tree.pkl
├── monotonic_clustering.pkl
├── gmm_model.pkl
└── feature_metadata.pkl
```

### Prediction Pipeline

Generate predictions for all products:
```bash
python scripts/predict.py
```

**What happens during prediction:**
1. Loads raw data (ALL rows)
2. Preprocesses data (adjusts violations, no filtering)
3. Engineers features
4. Scales using saved RobustScaler
5. Generates predictions from all 3 models
6. Ensembles via majority voting
7. Applies distribution reallocation to match target (10% KVI, 30% SD, 60% PG)
8. Saves final predictions to `data/results/final_predictions.csv`

**Expected output:**
```
data/results/final_predictions.csv

Columns:
- product_id: Product identifier
- predicted_label: KVI, SD, or PG
```

### Testing

Validate trained models:
```bash
python scripts/test_train.py
```

## Data Preprocessing

### Training Mode (train.py)
- **Filters violations**: Removes rows with:
  - `price_elasticity > 0` (violates law of demand)
  - `total_units_sold < 0` (negative quantities)
  - `total_revenue < 0` (impossible transactions)
- **Removes duplicates**: Keeps first occurrence
- **Converts types**: `total_units_sold` → integer

### Prediction Mode (predict.py)
- **Adjusts violations**: Caps invalid values but keeps all rows:
  - `price_elasticity > 0` → 0
  - `total_units_sold < 0` → 0
  - `total_revenue < 0` → 0
- **No filtering**: All rows retained for prediction

## Feature Engineering

**Original features (9):**
- price_elasticity, total_online_views, total_units_sold, total_revenue
- median_basket_position, mean_added_revenue_to_basket, mean_basket_profit
- competitive_price_range, price_volatility

**Engineered features (11):**
1. `revenue_per_unit` = total_revenue / total_units_sold
2. `conversion_rate` = total_units_sold / total_online_views
3. `basket_value_ratio` = mean_added_revenue_to_basket / total_revenue
4. `profit_margin_proxy` = mean_basket_profit / mean_added_revenue_to_basket
5. `competitive_pressure` = competitive_price_range / revenue_per_unit
6. `price_stability` = 1 / (price_volatility + 0.001)
7. `sales_velocity` = conversion_rate × median_basket_position
8. `elasticity_revenue_interaction` = price_elasticity × total_revenue
9. `volatility_competition_interaction` = price_volatility × competitive_price_range
10. `early_basket_indicator` = 1 if median_basket_position ≤ 2 else 0
11. `basket_profit_lift` = mean_basket_profit - revenue_per_unit

**Total features:** 20 (9 original + 11 engineered)

## Model Architecture

### 1. Decision Tree
- **Purpose**: Supervised baseline, trained on 54 labeled samples
- **Parameters**: max_depth=3, max_features=4, min_samples_split=5
- **Features used**: All 20 features
- **Output**: Top 4 most important features for clustering

### 2. Hierarchical Clustering
- **Purpose**: Monotonic clustering on unlabeled data
- **Parameters**: linkage='ward', metric='euclidean'
- **Features used**: Top 4 from Decision Tree
- **Prediction**: KNN (n_neighbors=5) trained on cluster assignments
- **Mapping**: Competitive pressure-based (low→PG, mid→SD, high→KVI)

### 3. Gaussian Mixture Model (GMM)
- **Purpose**: Hyperparameter-optimized clustering
- **Parameters**: covariance_type='full', n_init=10, max_iter=200
- **Features used**: Top 4 from Decision Tree
- **Mapping**: Competitive pressure-based (low→PG, mid→SD, high→KVI)

### Ensemble Strategy
- **Voting**: Mode of 3 model predictions
- **Reallocation**: Adjust ensemble to match target distribution (10% KVI, 30% SD, 60% PG)
  - Uses cluster distances from GMM
  - Reallocates samples closest to deficit segment clusters

## Expected Segment Characteristics

### KVI (Key Value Items) - 10%
- High price elasticity (sensitive to price changes)
- High competitive pressure
- Low median basket position (purchased early)
- Low conversion rate

### SD (Sales Drivers) - 30%
- High total units sold
- High conversion rate
- Moderate revenue per unit
- Moderate competitive pressure

### PG (Profit Generators) - 60%
- High revenue per unit
- High profit margin proxy
- Low price elasticity (inelastic, can sustain higher prices)
- Low competitive pressure
- Low sales velocity

## Configuration

### Adjustable Parameters

**In `scripts/train.py`:**
```python
DATA_PATH = "data/raw/data.parquet"  # Input data path
MODELS_DIR = "models"                 # Model output directory
RANDOM_STATE = 42                     # Reproducibility seed
```

**In `scripts/predict.py`:**
```python
DATA_PATH = "data/raw/data.parquet"  # Input data path
TARGET_DISTRIBUTION = {               # Desired segment proportions
    'KVI': 0.10,
    'SD': 0.30,
    'PG': 0.60
}
```

**Outlier detection:**
```python
# In train.py, line ~110
contamination=0.05  # Expected outlier proportion (5%)
```

## Output Format

**`data/results/final_predictions.csv`:**
```csv
product_id,predicted_label
1,KVI
2,SD
3,PG
...
```

## Troubleshooting

### Issue: FileNotFoundError
```
FileNotFoundError: Data file not found: data/raw/data.parquet
```
**Solution**: Ensure data file exists at specified path. Check `DATA_PATH` in train.py/predict.py.

### Issue: Model files not found
```
FileNotFoundError: Model file not found: models/robust_scaler.pkl
```
**Solution**: Run `python scripts/train.py` before `python scripts/predict.py`.

### Issue: Feature mismatch
```
ValueError: X has 19 features, but model expects 20
```
**Solution**: Retrain models. Feature engineering may have changed.

### Issue: All predictions same segment
**Solution**: Check target distribution. Reallocation requires diverse initial predictions.


## Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning models (Decision Tree, Hierarchical, GMM, Isolation Forest)
- **scipy**: Statistical functions
- **pyarrow**: Parquet file reading

## Author

Jakub Drahokoupil
Prague 26-10-2025


## References

- Notebook: `notebooks/Home_assignment.ipynb`
- Assignment: `DS_Home_Assignment.pdf`