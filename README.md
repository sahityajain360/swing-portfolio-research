# Quantitative Swing Portfolio Optimization System
### Multi-stage ML ensemble for Nifty 500 equity selection · 15-year walk-forward validated · Python

> **Research paper in progress.** This repository contains the public methodology,
> analysis scripts, and results from an independent quantitative finance research project.
> Core feature engineering and model weights are proprietary and not included.

---

## Research Summary

A seven-phase research project building a systematic portfolio selection system for
Indian equities. Starting from a naive Ridge stacker and iterating through deep learning,
graph networks, NLP integration, and a rigorous leakage audit — culminating in three
defensible final strategies with institutional-quality risk metrics.

**Universe:** Nifty 500 | **Horizon:** 20-day rebalancing, top-25 portfolio
**Period:** 2011–2026 (15 years, ~1.34M daily rows) | **Broker model:** Zerodha (exact fee structure)
**Validation:** Strict walk-forward — train on years < test_year, test on test_year

---

## Final Results (Leakage-Free, Out-of-Sample 2014–2025)

| Strategy | CAGR | Sharpe | Max Drawdown | Description |
|----------|------|--------|--------------|-------------|
| **Alpha Engine** | **71.25%** | **1.645** | **-21.89%** | 17-signal XGBoost + 10% MAE stop-loss |
| **Institutional Stabilizer** | **40.03%** | **1.685** | **-18.01%** | 17-signal PyTorch MMoE + 10% MAE stop-loss |
| **SOTA Hybrid** | **66.68%** | **1.720** | **-13.68%** | Dynamic Macro Routing (XGB ↔ MMoE) |

> Zerodha transaction costs modelled throughout: COST\_BUY = 0.1687%, COST\_SELL = 0.1537%

**Theoretical ceiling (Perfect-Foresight Oracle):** 1554% CAGR | 6.617 Sharpe | -3.29% Max DD

---

## Research Phases

### Phase 1 — Baseline & Oracle Benchmark
Established a 9-model Ridge stacker baseline (46.42% CAGR / 1.662 Sharpe).
Built a **Perfect-Foresight Oracle** selecting the literal top-25 stocks by
actual future 20-day return each period — revealing the theoretical ceiling and the
true alpha gap our system needs to close.

**Key finding:** Our base models shared only ~2.5/25 stocks with the Oracle per period (10% overlap).
The return gap between rank-25 and rank-26 is just 0.31% — predicting the exact boundary
is mathematically noise-limited; a realistic target is 4–5 Oracle stocks per period.

---

### Phase 2 — Oracle DNA Analysis
Used **Cohen's d** to systematically measure which features statistically separate
Oracle picks from non-Oracle picks across 15 years of rebalance dates.

**Top discriminating features (Cohen's d):**

| Feature | Cohen's d | Direction | Meaning |
|---------|-----------|-----------|---------|
| `std_60_rank` | +0.438 | ↑ | Oracle selects **high-volatility** stocks |
| `dist_52w_low_z` | +0.361 | ↑ | Far from 52-week low |
| `MA200_rank` | **−0.308** | ↓ | Oracle picks stocks **below** long-term MA |
| `MA20_rank` | **−0.291** | ↓ | Short-term pullback confirmed |
| `dollar_vol_20_rank` | −0.224 | ↓ | Avoids large-cap liquid names |

**Oracle DNA profile:** High-volatility mid/small-cap with strong long-term momentum
but short-term pullback — a systematic "buy the dip in an uptrend" strategy.

**Surprising null result:** Volume surge (Vol\_Ratio\_5d) showed Cohen's d ≈ −0.009.
Oracle stocks had *slightly lower* volume than average — volume is not predictive.

---

### Phase 3 — 17-Signal Architecture & Graph Network

Expanded from 9 to 17 signals to address a **sector rotation blindspot**
(Capital Goods, Metals, and Power sectors were systematically missed):

| Signal Group | Models | Purpose |
|---|---|---|
| Stage 1: Regressors | CAT, LGB, XGB | Predict risk-adjusted 20-day return |
| Stage 2: Rankers | CAT, LGB, XGB | NDCG/YetiRank cross-sectional ranking |
| Stage 3: Classifiers | CAT, LGB, XGB | Above-median binary label |
| Top-Decile Classifiers | CAT, LGB, XGB | Top-10% binary label |
| Top-10 Absolute | CAT, LGB, XGB | Literal top-10 stocks by return |
| STHAN Graph Network | KNN-GAT | Spatiotemporal capital flow between sectors |
| Oracle Classifier V2 | XGB | Probability of being a true Oracle pick |

**STHAN (Spatiotemporal Heterogeneous Attention Network):**
A KNN-based dynamic graph where each stock is a node, edges connect the K=5 most
correlated peers per rebalance date, and a 2-layer GAT with LayerNorm propagates
capital flow signals across the graph. Trained strictly walk-forward.

---

### Phase 4 — Risk Management & The Stop-Loss Discovery

Tested two risk management approaches:

| Method | CAGR | Sharpe | Max DD |
|--------|------|--------|--------|
| Inverse Volatility Sizing only | 65.55% | 1.368 | -46.67% |
| **10% Hard Stop-Loss only** | **99.27%** | **2.081** | **-17.43%** |
| Combined | 98.68% | 1.997 | -17.02% |

**Why inverse volatility failed:** Oracle DNA explicitly showed the system should
*target* high-volatility stocks. Inverse vol sizing mathematically penalizes exactly
the stocks that carry the most alpha — a fundamental contradiction.

**Why top-down macro filters failed:** A Random Forest macro regime filter correctly
avoided 2018 (-12% DD reduction) but misidentified 2020 and 2022, cutting exposure
during the fastest recoveries. Bottom-up stop-losses dominate because they respond
to actual stock-level evidence, not predicted macro regimes.

---

### Phase 5 — Deep Learning: MMoE Architecture

Replaced the XGBoost meta-ranker with a **PyTorch Multi-gate Mixture-of-Experts**:

```
Input: 17 signal percentiles + 5 macro features
         ↓
Gate Network: Linear(21→32) → ReLU → Linear(32→3) → Softmax
         ↓                    ↓                    ↓
   Expert 1            Expert 2            Expert 3
Linear(21→64)       Linear(21→64)       Linear(21→64)
   → ReLU              → ReLU              → ReLU
   → Drop(0.2)         → Drop(0.2)         → Drop(0.2)
Linear(64→16)       Linear(64→16)       Linear(64→16)
   → ReLU              → ReLU              → ReLU
Linear(16→1)        Linear(16→1)        Linear(16→1)
         ↓                    ↓                    ↓
         └──────── Weighted Sum (gate weights) ────┘
                             ↓
                      Final prediction
```

**Macro gating features:** Market\_Mom\_60d, Market\_DD\_252d, Breadth\_200DMA,
Dispersion\_20d, Up\_Down\_Vol\_Ratio — all standardized with strict walk-forward
statistics (train mean/std only, applied to test).

**MMoE result:** Sacrificed the 99% CAGR bull-run upside to achieve smooth equity
curves and elite drawdown protection. Acts as a defensive institutional stabilizer
rather than an aggressive alpha engine.

---

### Phase 6 — The Leakage Audit (Critical)

A line-by-line audit found **two forms of subtle lookahead bias** inflating pre-audit Sharpes above 2.0:

**Bug 1 — The Stop-Loss Trap:**
Original implementation: if `Target_Raw < -10%`, cap return at -10%.
Problem: `Target_Raw` is the 20-day period return. A stock that fell to -15%
intra-month but recovered to -5% by day 20 would *not* trigger the stop in reality,
but the original code would *not* apply the stop either — because it checked the final
return, not the intra-month minimum.

Fix: Computed **Maximum Adverse Excursion (MAE)** — the minimum close price
observed across the next 20 trading days, divided by entry close. If
`Forward_20d_Min_Ret ≤ −10%`, the actual return is capped at −10% regardless of
recovery. This is the true, conservative stop-loss calculation.

**Bug 2 — Global Macro Standardization:**
Original: standardize macro features across the entire dataset before walk-forward splits.
Problem: 2020 COVID volatility was leaking into 2014 standardization parameters.

Fix: Macro features are now standardized using only training-set statistics at each
walk-forward fold boundary.

**Impact of leakage audit:**

| Model | Pre-Audit Sharpe | Post-Audit Sharpe | CAGR Change |
|-------|-----------------|-------------------|-------------|
| XGB Champion | ~2.3 | 1.645 | 99% → 71% |
| MMoE Network | ~2.0 | 1.685 | 54% → 40% |
| Dynamic Routing | ~2.1 | 1.720 | 90% → 67% |

---

### Phase 7 — NLP: FinBERT Sentiment Integration

**Data sources:**
- Times of India archive: 3.8M rows, filtered to ~147,516 business news articles
- Business Standard: ~50K financial news rows

**Processing pipeline:**
- Regex mapping of article text to Nifty 500 tickers
- GPU-batched inference through `ProsusAI/finbert` (HuggingFace)
- Strict T−1 shift applied to all NLP features (no market-close lookahead)

**Features generated:**
- `Macro_Sent_7d` / `Macro_Sent_Shock` — systemic economy-level sentiment
- `Micro_Sent_7d` / `Micro_Sent_Shock` / `Micro_News_Vol` — ticker-level sentiment

**Result:** `Macro_Sent_Shock` ranked **9th out of 130+ features** by importance.
None of the Micro\_Sent features appeared in the top 15.

**NLP sparsity finding (original empirical result):**

| Metric | Value |
|--------|-------|
| Total stock-days in dataset | 1,351,383 |
| Stock-days with any idiosyncratic news | 5,554 |
| **Dataset sparsity** | **99.59%** |
| Tickers with zero articles across 12 years | 73.9% (362/490) |
| Most-covered ticker | OIL: 2,686 articles (likely false positives from language) |

**Conclusion:** Open-source newspaper archives are inadequate for idiosyncratic NLP
alpha in Indian mid/small-cap equities. Aggregated systemic macro sentiment works;
per-ticker sentiment does not. Premium data (Bloomberg, Reuters, earnings transcripts)
would be required for viable micro-sentiment signals.

---

### Phase 8 — Dynamic Macro Routing (Final SOTA)

A Random Forest trained walk-forward on macro features predicts the "Alpha Pool"
(Oracle minus market return). When Alpha Pool is predicted High (≥60th percentile),
capital routes 80% XGB / 20% MMoE. When Low, routes 20% XGB / 80% MMoE.

```
Market State Features → RF Regressor → Alpha Pool Prediction
         ↓
    High Alpha?  ──Yes──→  80% XGB  +  20% MMoE
         │
         No
         │
         └──────────────→  20% XGB  +  80% MMoE
```

This combines the 71% CAGR upside of the XGBoost engine with the -18% Max DD
protection of the MMoE stabilizer, achieving -13.68% Max DD — the best risk-adjusted
profile of all three configurations.

---

## Oracle vs System Analysis

```
Oracle CAGR (no costs):     1904.75%
System CAGR (no costs):       56.99%
Capture Ratio:                 2.99%
Avg overlap per period:     2.5 / 25 stocks
Avg alpha gap per period:      23.65%
```

**Where we miss Oracle picks (most enriched sectors):**

| Sector | Miss Enrichment |
|--------|----------------|
| Metals & Mining | 1.60× |
| Capital Goods | 1.36× |
| Chemicals | 1.28× |
| Power | 1.16× |

These sectors exhibit the strongest regime-dependent sector rotation — they appear
in the Oracle predominantly during specific macro windows that our cross-sectional
models do not fully capture, motivating the STHAN graph network architecture.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data | NSE OHLCV, Times of India archive, Business Standard |
| Features | pandas, NumPy (123 engineered features) |
| Stage Models | CatBoost, LightGBM, XGBoost (Regressors, Rankers, Classifiers) |
| Meta-Learner | XGBoost Ranker (YetiRank / NDCG) |
| Deep Learning | PyTorch (MMoE), PyTorch Geometric (KNN-GAT) |
| NLP | HuggingFace `ProsusAI/finbert`, GPU-batched |
| Routing | scikit-learn RandomForest |
| Validation | Custom walk-forward engine |
| Language | Python 3.10+ |

---

## Repository Structure

```
swing-portfolio-research/
├── README.md
├── analysis/
│   ├── oracle_dna_analysis.py       # Cohen's d DNA methodology
│   ├── oracle_vs_system.py          # Oracle-system overlap diagnostic
│   └── nlp_sparsity_check.py        # NLP coverage audit (original finding)
└── assets/
    └── architecture.png
```

> **Note:** Feature engineering, model training pipelines, and signal logic are
> proprietary and not included in this repository. This is standard practice in
> quantitative finance research.

---

## Key Research Contributions

1. **Oracle DNA methodology** — systematic use of Cohen's d across 130+ features to
   reverse-engineer what drives true alpha in Nifty 500 equities

2. **MAE stop-loss vs. terminal return stop-loss** — empirical demonstration that
   checking terminal 20-day return (common in academic backtests) substantially
   overstates stop-loss effectiveness compared to true Maximum Adverse Excursion

3. **NLP sparsity finding** — original empirical measurement showing 99.59% sparsity
   in open-source Indian financial news at the stock-day level, with implications for
   NLP alpha research in emerging markets

4. **MMoE routing collapse** — the gating network allocates near-uniformly across
   experts when base models share identical feature spaces, representing a meaningful
   negative result for adaptive neural ensembling

---

## About

Independent quantitative finance research conducted at Manipal Institute of Technology
(B.Tech CSE AI & ML, 2023–27) under academic supervision.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-sahitya360-blue)](https://linkedin.com/in/sahitya360/)
[![GitHub](https://img.shields.io/badge/GitHub-sahityajain360-black)](https://github.com/sahityajain360)
