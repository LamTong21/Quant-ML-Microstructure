# 📈 Emerging Equity Predictability: Microstructure & Kinematics Framework

> **Paper Reference:** *From Static Price Topology to Order-Flow Microstructure and Regime-Conditioned Kinematics: An Econometrically Grounded Machine Learning Framework for Emerging Equity Predictability.*
> 

## 📌 Executive Summary

Applying pure Machine Learning classifiers to raw OHLCV candlestick patterns in emerging frontier markets often suffers from severe **alpha decay**, **look-ahead bias** (from commercial backward price adjustments), and **friction erosion** in real-world trading.

This repository implements a rigorous **5-Stage Econometric-ML Framework** that transitions from static price topologies to **intraday order-flow dynamics (OFI, VPIN)**, **higher-order kinematics (velocity, acceleration, jerk)**, and **adaptive GMM regime filters**. Designed specifically for institutional quantitative research, this pipeline prioritizes out-of-sample robustness and strict data leakage prevention over in-sample illusion.

## 🏛️ The 5-Stage System Architecture

$$Stage 1$$

Data Integrity & Dynamic Labeling

- **Forward-Adjustment Engine:** Eliminates corporate-action leakage by computing monotonic forward-adjusted prices (anchoring historical prices and projecting forward).
- **Dynamic Triple-Barrier Labeling:** Assigns target labels ($\{-1, 0, 1\}$) using localized EWMA volatility ($\sigma_{t, 20}$) to dynamically scale profit-taking and stop-loss barriers.

$$Stage 2$$

Extended DGP Diagnostics Sensor

- Operates purely on the In-Sample Train set to scan time-series characteristics:
    - **Distribution & Memory:** Jarque-Bera, DFA Hurst exponent, Lo-MacKinlay Variance Ratio.
    - **Cycles & Jumps:** Fast Fourier Transform (FFT) dominant cycles, Barndorff-Nielsen & Shephard Bipower Variation jump detection.
    - **Complexity:** Bandt-Pompe Permutation Entropy, BDS Test.

$$Stage 3$$

Evidence-Based Institutional Feature Factory

- Dynamically routes and generates features based on Stage 2 diagnostic payloads:
    - **Microstructure:** Intraday Order Flow Imbalance (OFI), VPIN proxy, Corwin-Schultz spread, Amihud illiquidity.
    - **Kinematics:** First-order momentum (Velocity), Second-order (Acceleration), Third-order (Jerk), and Squeeze ratios.
    - **Wavelets & Stress:** Haar Wavelet Multiresolution ($d_1, d_2$), Semi-Variance Asymmetry, Drawdown Velocity, Rolling GMM Bull/Bear probabilities.

$$Stage 4$$

Multi-Stage Statistical Pruning

- An independent, model-free hypothesis testing chain ensuring orthogonal and causal feature spaces:
    1. **Redundancy Control:** Linear VIF filter ($VIF \le 5.0$) + Non-linear Hierarchical Risk Parity (HRP) clustering for Medoid selection.
    2. **Causality Screen:** Dual testing via Granger Causality F-test and Lagged Transfer Entropy (Mutual Information Proxy).
    3. **Kinematic Transform:** Volatility-scaled lag transformations at statistically optimal delays ($L^*$).
    4. **Information Theory:** Regime-Conditioned Mutual Information filter.

$$Stage 5$$

Purged Walk-Forward ML Engine & Realistic Execution

- **Validation:** 5-Fold Expanding Walk-Forward with a `5-day purging gap` and `60-day warm-up buffer` to absolutely prevent target overlap.
- **Modeling:** Multi-class XGBoost (`multi:softprob`) optimized via Nested RandomizedSearchCV.
- **Execution Frictions:** $t+1$ execution delay, 20 bps nominal transaction costs (commission + slippage), high-confidence thresholding ($P > 0.55$), and an Adaptive GMM Regime Hard Filter for capital preservation.

## 📊 Empirical Out-of-Sample Results

**Target Asset:** DIG (Vietnam Stock Exchange - HOSE) | **Period:** 2018–2026 | **Observations:** 1,996 trading sessions.

The upgraded framework (`Modeling02`) demonstrates a massive leap in realistic trading performance compared to the traditional static daily OHLCV approach (`Modeling01`).

| **Metric** | **Modeling01 (Baseline)** | **Modeling02 (Upgraded)** | **Improvement (Δ)** |

| **Rank IC** | +0.0286 | **+0.0425** | **+48.6%** (Raw predictive power) |

| **Total Trades (**$N_{trades}$**)** | 232.0 | **103.0** | **-55.6%** (Noise reduction) |

| **Total Turnover** | 464.0 | **206.0** | **-55.6%** (Friction mitigation) |

| **Max Drawdown (MDD)** | -73.04% | **-44.40%** | **+28.64%** (Capital preservation) |

| **Net Annualized Return** | -15.82% | **-3.31%** | **+12.51%** |

| **Net Sharpe Ratio** | -0.6390 | **-0.1925** | **+0.4465** |

*(Note: The strategy is strictly evaluated **Net of Fees** with a* $t+1$ *execution lag constraint).*

### Visualizations

*Please refer to the `docs/figures/` directory for generated plots.*

- `equity_curve.png`: Cumulative Return vs. Buy & Hold Benchmark.
- `feature_importance.png`: Top features driving Alpha (dominated by *Skewness Dynamics*, *VPIN Momentum*, and *Roll Measure*).

## 📂 Repository Structure

```
emerging-equity-microstructure-kinematics/
├── .github/workflows/        # CI/CD pipelines (Automated PyTest & Flake8)
├── configs/                  # YAML configurations (Hyperparams, Assets, Thresholds)
├── data/                     # Raw and Processed data storage (Ignored in Git)
├── docs/                     # Paper PDF and Figures/Plots
├── notebooks/                # Jupyter Notebooks for interactive research & EDA
├── src/                      # Core Pipeline Source Code
│   ├── data_loader/          # Stage 1: Integrity, Return Topology, Triple-Barrier
│   ├── dgp_diagnostics/      # Stage 2: Time-Series DGP Sensors
│   ├── feature_factory/      # Stage 3: Modular Evidence-based Feature Generation
│   ├── feature_selection/    # Stage 4: Statistical Pruning (VIF, HRP, Granger, MI)
│   ├── engine/               # Stage 5: Purged CV, XGBoost, Backtest Simulator
│   └── pipeline.py           # End-to-end execution script
├── tests/                    # Unit tests for data integrity and causality logic
├── environment.yml           # Conda environment specs
├── requirements.txt          # Python package dependencies
└── README.md
```

## 🚀 Quickstart

**1. Clone the repository:**

```
git clone https://github.com/your-username/emerging-equity-microstructure-kinematics.git
cd emerging-equity-microstructure-kinematics
```

**2. Set up the environment (using Conda):**

```
conda env create -f environment.yml
conda activate emerging-equity-predictability
```

*(Alternatively, use `pip install -r requirements.txt` in your standard virtual environment).*

**3. Run the complete pipeline:**

```
python src/pipeline.py --config configs/asset_config.yaml
```

**4. Run Automated Tests:**

```
pytest tests/ -v
```

## 🔮 Future Work

1. **Level 2 Limit Order Book (L2 LOB):** Directly ingest real-time L2 depth to calculate granular Order Book Imbalance, Replenishment, and Cancellation rates instead of relying on $15m$ intraday proxies.
2. **Cross-Sectional Portfolio Optimization:** Expand the single-asset focus to the entire VN30 basket, coupled with Hierarchical Risk Parity (HRP) portfolio allocation.
3. **Sequential Deep Learning:** Feed the orthogonalized feature matrix outputted by Stage 4 into Temporal Fusion Transformers (TFT) or LSTM-Attention networks for capturing complex non-linear sequence dependencies.

## 📜 License

This project is licensed under the MIT License.

*Disclaimer: This repository is for academic and quantitative research purposes only. It does not constitute financial advice.*