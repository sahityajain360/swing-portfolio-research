"""
oracle_dna_analysis.py
======================
Oracle DNA Analysis — What separates the top-25 stocks from the rest?

This script uses Cohen's d effect size to systematically measure which
features statistically discriminate Oracle picks (perfect-foresight top-25
by actual future 20-day return) from non-Oracle picks across all rebalance
periods.

The "Oracle" is a perfect-foresight benchmark: at each rebalance date it
selects the exact 25 stocks with the highest actual forward 20-day return.
It represents the theoretical maximum performance achievable on this universe.

Key findings from running this on Nifty 500 (2011–2026):
  - Oracle DNA: high-volatility + long-term momentum + short-term pullback
  - Volume surge has near-zero Cohen's d (surprise null result)
  - Sector concentration is highly regime-dependent
  - Oracle shares only ~2.5/25 stocks (~10%) with our system per period
  - The return gap between rank-25 and rank-26 is ~0.31% — noise-bound

This file contains NO proprietary features, model weights, or trading logic.
It is a pure analytical tool operating on any OHLCV-derived feature parquet.
"""

import pandas as pd
import numpy as np
import os


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — update these paths for your environment
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH = "path/to/your/nifty500_features.parquet"
INDUSTRY_CSV = "path/to/ind_nifty500list.csv"
OUTPUT_FOLDER = "path/to/output_folder"

# Columns that are never features (raw prices, identifiers, targets)
NON_FEATURE_COLS = {
    "Date", "Ticker", "Open", "High", "Low", "Close", "Volume",
    "Target", "Target_Raw", "Industry", "Year", "Regime",
    "Oracle_Label", "Oracle_Prob", "Forward_20d_Return",
}

# Portfolio size and rebalancing frequency
TOP_K = 25
REBALANCE_EVERY_N_DAYS = 20


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: COHEN'S D EFFECT SIZE
# ──────────────────────────────────────────────────────────────────────────────
def cohens_d(group_a: pd.Series, group_b: pd.Series) -> float:
    """
    Computes Cohen's d effect size between two groups.

    Cohen's d = (mean_A - mean_B) / pooled_std

    Interpretation:
        |d| < 0.2  →  negligible
        |d| = 0.2  →  small
        |d| = 0.5  →  medium
        |d| = 0.8  →  large

    A positive d means group_A (Oracle picks) has higher values.
    A negative d means Oracle picks have lower values (e.g., below their MA).

    Args:
        group_a: Feature values for Oracle picks (positive class)
        group_b: Feature values for non-Oracle picks (negative class)

    Returns:
        float: Cohen's d effect size
    """
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return 0.0

    mean_a, mean_b = group_a.mean(), group_b.mean()
    var_a = group_a.var(ddof=1)
    var_b = group_b.var(ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    if pooled_std < 1e-9:
        return 0.0

    return float((mean_a - mean_b) / pooled_std)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: ORACLE LABELING
# ──────────────────────────────────────────────────────────────────────────────
def label_oracle(df: pd.DataFrame, top_k: int = 25, every_n: int = 20) -> pd.DataFrame:
    """
    Labels each row as Oracle (1) or non-Oracle (0) based on actual future return.

    The Oracle is applied every `every_n` trading days, selecting the top_k
    stocks by true forward return (Target_Raw column).

    Args:
        df:     DataFrame with columns [Date, Ticker, Target_Raw, features...]
        top_k:  Portfolio size (default 25)
        every_n: Rebalancing frequency in trading days (default 20)

    Returns:
        DataFrame with added Oracle_Top25 column (binary 0/1)
    """
    df = df.copy()
    df["Oracle_Top25"] = 0

    unique_dates = sorted(df["Date"].unique())
    rebalance_dates = unique_dates[::every_n]

    print(f"Labeling Oracle on {len(rebalance_dates)} rebalance dates...")

    for dt in rebalance_dates:
        day_mask = df["Date"] == dt
        day_data = df[day_mask]
        if len(day_data) < top_k * 2:
            continue
        top_idx = day_data.nlargest(top_k, "Target_Raw").index
        df.loc[top_idx, "Oracle_Top25"] = 1

    labeled_count = df["Oracle_Top25"].sum()
    print(f"Total Oracle rows labeled: {labeled_count:,}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: FEATURE DNA (Cohen's d for every feature)
# ──────────────────────────────────────────────────────────────────────────────
def compute_oracle_dna(df: pd.DataFrame, feature_cols: list, top_n: int = 20) -> pd.DataFrame:
    """
    Computes Cohen's d for every feature, separating Oracle vs non-Oracle rows.

    Args:
        df:           DataFrame with Oracle_Top25 column and feature columns
        feature_cols: List of feature column names to test
        top_n:        How many features to display in ranked output

    Returns:
        pd.DataFrame sorted by |Cohen's d|, columns: [feature, cohens_d, direction, meaning]
    """
    oracle_rows = df[df["Oracle_Top25"] == 1]
    rest_rows = df[df["Oracle_Top25"] == 0]

    print(f"\nOracle rows: {len(oracle_rows):,} | Non-Oracle rows: {len(rest_rows):,}")

    results = []
    for col in feature_cols:
        a = oracle_rows[col].dropna()
        b = rest_rows[col].dropna()
        d = cohens_d(a, b)
        direction = "↑ Oracle higher" if d > 0 else "↓ Oracle lower"
        results.append({"feature": col, "cohens_d": d, "abs_d": abs(d), "direction": direction})

    dna_df = pd.DataFrame(results).sort_values("abs_d", ascending=False).reset_index(drop=True)
    dna_df = dna_df.drop(columns=["abs_d"])

    print(f"\n{'='*65}")
    print(f"ORACLE DNA — Top {top_n} Features by Cohen's d")
    print(f"{'='*65}")
    print(f"{'Rank':<5} {'Feature':<30} {'d':>8}  {'Direction'}")
    print("-" * 65)
    for i, row in dna_df.head(top_n).iterrows():
        print(f"{i+1:<5} {row['feature']:<30} {row['cohens_d']:>8.4f}  {row['direction']}")

    return dna_df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: ORACLE PERSISTENCE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
def analyze_persistence(df: pd.DataFrame, every_n: int = 20) -> dict:
    """
    Measures how many Oracle stocks repeat between consecutive periods.

    Low persistence = Oracle is mostly fresh each time → momentum-based
    persistence features have weak predictive value.

    Args:
        df: DataFrame with [Date, Ticker, Oracle_Top25]

    Returns:
        dict with mean and std of overlap fraction between consecutive periods
    """
    unique_dates = sorted(df["Date"].unique())
    rebalance_dates = unique_dates[::every_n]

    overlaps = []
    prev_oracle = set()

    for dt in rebalance_dates:
        day_data = df[df["Date"] == dt]
        curr_oracle = set(day_data[day_data["Oracle_Top25"] == 1]["Ticker"])
        if prev_oracle:
            overlap = len(prev_oracle & curr_oracle) / len(prev_oracle) if prev_oracle else 0
            overlaps.append(overlap)
        prev_oracle = curr_oracle

    result = {
        "mean_persistence": np.mean(overlaps),
        "std_persistence": np.std(overlaps),
        "min_persistence": np.min(overlaps),
        "max_persistence": np.max(overlaps),
    }
    print(f"\n--- Oracle Persistence ---")
    print(f"Mean: {result['mean_persistence']:.2%} ± {result['std_persistence']:.2%}")
    print(f"Range: {result['min_persistence']:.2%} – {result['max_persistence']:.2%}")
    print("Interpretation: Low persistence means Oracle picks are mostly fresh each period.")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5: ORACLE RETURNS BY MARKET REGIME
# ──────────────────────────────────────────────────────────────────────────────
def analyze_regime_returns(df: pd.DataFrame, every_n: int = 20) -> pd.DataFrame:
    """
    Measures Oracle return by market regime (Bull / Flat / Bear).

    Regime is defined by the cross-sectional mean 20-day return on each date:
        Bull:  market mean > +2%
        Bear:  market mean < -2%
        Flat:  otherwise

    Args:
        df: DataFrame with [Date, Ticker, Target_Raw, Oracle_Top25]

    Returns:
        pd.DataFrame with regime-level Oracle return statistics
    """
    unique_dates = sorted(df["Date"].unique())
    rebalance_dates = unique_dates[::every_n]

    records = []
    for dt in rebalance_dates:
        day_data = df[df["Date"] == dt]
        oracle_picks = day_data[day_data["Oracle_Top25"] == 1]
        if oracle_picks.empty:
            continue
        market_ret = day_data["Target_Raw"].mean()
        oracle_ret = oracle_picks["Target_Raw"].mean()
        regime = "Bull" if market_ret > 0.02 else ("Bear" if market_ret < -0.02 else "Flat")
        records.append({"Date": dt, "Market_Ret": market_ret, "Oracle_Ret": oracle_ret, "Regime": regime})

    regime_df = pd.DataFrame(records)
    summary = regime_df.groupby("Regime")["Oracle_Ret"].agg(["mean", "std", "count"])
    summary.columns = ["Mean Oracle Return", "Std", "Periods"]

    print(f"\n--- Oracle Returns by Market Regime ---")
    print(summary.to_string())
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6: SECTOR CONCENTRATION
# ──────────────────────────────────────────────────────────────────────────────
def analyze_sector_concentration(df: pd.DataFrame, every_n: int = 20) -> pd.DataFrame:
    """
    Measures which sectors are over-represented in Oracle picks vs the universe.

    Enrichment ratio = (Oracle sector fraction) / (Universe sector fraction)
    Ratio > 1.0 means Oracle over-selects that sector.

    Args:
        df: DataFrame with [Date, Ticker, Industry, Oracle_Top25]

    Returns:
        pd.DataFrame with sector enrichment ratios by year
    """
    if "Industry" not in df.columns:
        print("Industry column not found — skipping sector analysis")
        return pd.DataFrame()

    unique_dates = sorted(df["Date"].unique())
    rebalance_dates = unique_dates[::every_n]

    records = []
    for dt in rebalance_dates:
        day_data = df[df["Date"] == dt]
        oracle_picks = day_data[day_data["Oracle_Top25"] == 1]
        if oracle_picks.empty:
            continue

        universe_frac = day_data["Industry"].value_counts(normalize=True)
        oracle_frac = oracle_picks["Industry"].value_counts(normalize=True)

        for sector in oracle_frac.index:
            enrichment = oracle_frac[sector] / (universe_frac.get(sector, 1e-9))
            records.append({
                "Date": dt,
                "Year": pd.Timestamp(dt).year,
                "Sector": sector,
                "Enrichment": enrichment
            })

    sector_df = pd.DataFrame(records)
    annual = sector_df.groupby(["Year", "Sector"])["Enrichment"].mean().reset_index()
    top_by_year = annual.sort_values(["Year", "Enrichment"], ascending=[True, False])
    top3 = top_by_year.groupby("Year").head(3)

    print(f"\n--- Top 3 Over-Represented Sectors per Year ---")
    for year, g in top3.groupby("Year"):
        entries = [f"{row['Sector']} ({row['Enrichment']:.1f}×)" for _, row in g.iterrows()]
        print(f"  {year}: {' | '.join(entries)}")

    return annual


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  ORACLE DNA ANALYSIS")
    print("  Reverse-engineering perfect-foresight portfolio selection")
    print("=" * 65)

    # ── Load data ──────────────────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print(f"\n[DEMO MODE] Data file not found at: {DATA_PATH}")
        print("Generating synthetic data for demonstration...\n")

        # Synthetic fallback — runs without any real data
        np.random.seed(42)
        n = 50_000
        df = pd.DataFrame({
            "Date": pd.date_range("2015-01-01", periods=200, freq="B").repeat(250),
            "Ticker": [f"STOCK_{i%250:03d}" for i in range(n)],
            "Target_Raw": np.random.normal(0.03, 0.12, n),
            "std_60_rank": np.random.uniform(0, 1, n),
            "MA20_rank": np.random.uniform(0, 1, n),
            "MA200_rank": np.random.uniform(0, 1, n),
            "vol_ratio_5d": np.random.uniform(0.5, 2.0, n),
            "dist_52w_low_z": np.random.normal(0, 1, n),
            "Industry": np.random.choice(
                ["Financials", "IT", "Energy", "Metals", "Consumer", "Healthcare"], n
            ),
        })
    else:
        print(f"\nLoading data from: {DATA_PATH}")
        df = pd.read_parquet(DATA_PATH)

        if "Industry" not in df.columns and os.path.exists(INDUSTRY_CSV):
            ind_df = pd.read_csv(INDUSTRY_CSV)
            ind_map = ind_df.set_index("Symbol")["Industry"].to_dict()
            df["Industry"] = df["Ticker"].map(ind_map).fillna("Unknown")

        # Clip extreme returns for robustness
        if "Target_Raw" in df.columns:
            df["Target_Raw"] = df["Target_Raw"].clip(-0.95, 2.0)
        elif "Target" in df.columns:
            df["Target_Raw"] = df["Target"].clip(-0.95, 2.0)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # ── Identify feature columns (exclude non-features) ────────────────────────
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS and
                    pd.api.types.is_numeric_dtype(df[c])]
    print(f"Feature columns identified: {len(feature_cols)}")

    # ── Label Oracle ───────────────────────────────────────────────────────────
    df = label_oracle(df, top_k=TOP_K, every_n=REBALANCE_EVERY_N_DAYS)

    # ── Run all analyses ───────────────────────────────────────────────────────
    dna_df = compute_oracle_dna(df, feature_cols, top_n=20)
    persistence = analyze_persistence(df, every_n=REBALANCE_EVERY_N_DAYS)
    regime_stats = analyze_regime_returns(df, every_n=REBALANCE_EVERY_N_DAYS)
    sector_stats = analyze_sector_concentration(df, every_n=REBALANCE_EVERY_N_DAYS)

    # ── Save outputs ───────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    out_path = os.path.join(OUTPUT_FOLDER, "oracle_dna_results.parquet")
    dna_df.to_parquet(out_path)
    print(f"\nSaved DNA results to: {out_path}")

    print("\n" + "=" * 65)
    print("  ANALYSIS COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
