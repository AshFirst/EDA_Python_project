# EDA_Python_project

**Dataset:** [Gold Price Dynamics by krupalpatel07 on Kaggle](https://www.kaggle.com/datasets/krupalpatel07/gold-price-dynamics)

---

## How to Run

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels jupyter

# 2. Download dataset (requires Kaggle API key)
kaggle datasets download -d krupalpatel07/gold-price-dynamics
unzip gold-price-dynamics.zip

# 3. Launch notebook
jupyter notebook gold_price_analysis.ipynb
```

> **No data?** No problem. The notebook detects if the CSV is missing and auto-generates a statistically faithful synthetic dataset so every cell runs end-to-end.

---

## Section-by-Section Rationale

### 1. Environment Setup
We import every library up front so version conflicts surface immediately rather than mid-run. `warnings.filterwarnings('ignore')` keeps output clean for presentation; remove it during debugging. `sns.set_theme` and `plt.rcParams` ensure a consistent visual style throughout.

### 2. Data Loading
Financial CSVs from Kaggle often have inconsistent date formats. We use `parse_dates=['Date']` and immediately sort by date to guarantee temporal ordering — a requirement for any rolling or lag-based computation. The synthetic fallback uses **Geometric Brownian Motion (GBM)**, the standard model for equity/commodity price simulation, with parameters calibrated to real gold market characteristics.

### 3. Basic Exploration (`head` / `tail` / `describe` / `info`)
- **`head()` / `tail()`** — sanity-check that date ordering is correct and that no obvious data corruption exists at the boundary rows.
- **`describe(include='all')`** — reveals scale differences across columns (critical before scaling), the presence of NaNs via `count`, and categorical value distributions via `top` / `freq`.
- **`info(memory_usage='deep')`** — confirms dtypes and flags any columns read as `object` that should be numeric (a common Kaggle CSV issue).

### 4. Missing Value & Duplicate Analysis
We visualise missingness with both a bar chart (aggregate %) and a heatmap (pattern/location) because *random* missingness warrants a different strategy than *systematic* gaps (e.g., weekends are not in trading data).

**Imputation choice — forward-fill:** For time-series price data, using the previous day's value is the most economically sensible approach. Mean imputation would introduce future information into past rows and distort rolling-window features. We run a back-fill pass afterwards only to handle any leading NaNs that forward-fill cannot reach.

### 5. Feature Engineering & Encoding

| Feature | Rationale |
|---------|-----------|
| Year / Month / DayOfWeek / Quarter | Capture seasonal and cyclical patterns (e.g., gold often rallies in Q4 due to jewelry demand) |
| Spread (High − Low) | Intraday volatility proxy; higher spread = higher uncertainty |
| Daily Return / Log Return | Returns are more stationary than prices and model-friendly |
| MA-7 / MA-30 / MA-90 | Trend-following signals at different horizons; crossovers are key trading signals |
| Volatility (30-day rolling σ) | Annualised via √252; standard in finance for measuring risk |
| RSI (14-period) | Momentum oscillator; overbought >70, oversold <30 — canonical thresholds |
| MACD / Signal | Trend + momentum; histogram shows divergence strength |
| Target Direction | Binary label for classification task |

**Encoding choices:**
- `LabelEncoder` — shown for demonstration; assigns arbitrary integers.
- `OrdinalEncoder` with explicit order `['Bearish','Neutral','Bullish']` — *correct* for ordinal data since the order carries meaning.
- One-Hot Encoding for Month — avoids the model treating December (12) as "twice as important" as June (6).

### 6. Visualisations
- **Price + MA + Volume + Volatility** in one panel — the standard chart a quant analyst opens first.
- **Distribution histograms with normal overlay** — quick check for skewness and fat tails. Gold returns are known to be leptokurtic (heavier tails than normal), which affects risk modelling.
- **Box plots by Year/DayOfWeek** — reveal long-term price trends and any day-of-week effect.
- **Candlestick chart** — the most information-dense representation of OHLC data; standard in every trading platform.
- **RSI & MACD panel** — technical indicators are best interpreted visually alongside price.
- **Sentiment pie + count** — ensures the synthetic categorical variable is plausible and balanced.

### 7. Correlation Analysis
Pearson correlation quantifies *linear* relationships. The lower-triangular mask avoids redundancy. The pairplot on a 600-point sample reveals *non-linear* structure that correlation misses (e.g., gold/oil clustering). Key expected findings: SLV (silver) highly correlated with gold (same macro drivers), USD inversely correlated (gold is dollar-denominated).

### 8. Normalization & Scaling
All three scalers are shown side-by-side on the same column so the difference is concrete rather than abstract:
- **MinMaxScaler** is good for neural networks; bad when there are extreme outliers (a single spike compresses everything else).
- **StandardScaler** is the default for linear models and SVMs; assumes approximate normality.
- **RobustScaler** is the safest choice for financial data because extreme price events (flash crashes, etc.) are real, not errors, and should not distort the scale.

For all downstream models we use `StandardScaler` fitted *only on the training set* and applied to the test set — this is the correct approach to avoid data leakage.

### 9. Outlier Detection
Two methods are shown because they flag different things:
- **Z-score > 3** — identifies tails of the distribution; sensitive to the mean and std being themselves distorted by outliers.
- **IQR** — median-based; more robust. Standard boxplot rule (1.5 × IQR).

In financial data, "outliers" are usually **real market events** (e.g., COVID crash, 2008 GFC). They should be flagged and documented, not silently dropped. We leave them in for modelling.

### 10. Time-Series Decomposition & ADF Test
**Seasonal decomposition** (multiplicative model) splits the price series into interpretable components. Multiplicative is preferred over additive when the seasonal amplitude grows with the trend level — common in long commodity series.

**ADF test:** Most raw price series are non-stationary (unit root present, p > 0.05). After first-differencing (i.e., working with returns), the series typically becomes stationary. This is confirmed in the output and matters because non-stationary features can cause spurious regression results.

### 11. Machine-Learning Models

**Regression:** We predict the next day's Close price to compare model families:
- Linear models (LR, Ridge, Lasso) — establish a baseline; Ridge/Lasso add regularisation to combat multicollinearity between technical indicators.
- Tree ensembles (RF, GBM) — handle non-linearity and interactions automatically.
- SVR — strong for mid-size datasets with RBF kernel.
- KNN — non-parametric baseline; sensitive to scaling (hence StandardScaler is essential).

`shuffle=False` in the train/test split is critical — shuffling would leak future data into training.

**Classification:** Predicting price direction (up/down) is a harder, more realistic task. Random Forest is used as a single robust baseline. ~55–58 % accuracy is a reasonable result; consistently above 50 % is meaningful in practice.

### 12. Feature Importance & PCA
**Built-in RF importance** measures how much each feature reduces impurity across all trees — fast but can be biased toward high-cardinality features.

**Permutation importance** directly measures the drop in test performance when a feature is randomly shuffled — model-agnostic and more reliable.

**PCA** serves two purposes:
- The scree plot shows how many components capture 95 % of variance (useful for dimensionality reduction before deep learning).
- The 2-D scatter plot gives intuition for the structure in the feature space.

### 13. Summary
A concise table captures the most important findings so stakeholders can skim the results without re-running the notebook.

---

## File Outputs (generated by the notebook)

| File | Content |
|------|---------|
| `missing_values.png` | Missing data bar chart + heatmap |
| `price_history.png` | Price + MA + Volume + Volatility |
| `distributions.png` | Histograms with normal overlay |
| `boxplots.png` | Price by Year, Returns by DayOfWeek |
| `candlestick.png` | OHLC candlestick (last 120 days) |
| `indicators.png` | RSI + MACD technical analysis panel |
| `sentiment.png` | Sentiment distribution pie + bar |
| `correlation.png` | Pearson correlation heatmap |
| `pairplot.png` | Pairplot of core market variables |
| `scaling.png` | Scaler comparison histogram |
| `outliers.png` | Z-score and IQR outlier detection |
| `decomposition.png` | Seasonal decomposition (if statsmodels available) |
| `regression_results.png` | R² comparison + Actual vs Predicted |
| `confusion_matrix.png` | Classification confusion matrix |
| `feature_importance.png` | RF + permutation importance |
| `pca.png` | Scree plot + 2-D PCA scatter |

---

## Disclaimer
This notebook is for **educational and analytical purposes only**. Nothing in it constitutes financial advice or a trading signal.
