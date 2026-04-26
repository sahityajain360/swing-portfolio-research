"""
oracle_vs_system.py
===================
Oracle vs System Comparison — How close does our model get to perfect foresight?

This diagnostic tool compares the portfolios selected by a trained model
against the Perfect-Foresight Oracle (which always picks the exact top-25
stocks by actual forward return) across every rebalance period.

Key metrics computed:
  - Capture ratio (system CAGR / oracle CAGR)
  - Per-period stock overlap (how many Oracle picks did we get?)
  - Alpha gap (Oracle return minus system return)
  - Return distribution of missed picks vs wrong picks
  - Sector enrichment in missed Oracle stocks
  - Hardest periods (lowest overlap, biggest alpha gaps)

This analysis informed the 17-signal architecture expansion in Phase 3
of the research — specifically identifying Capital Goods, Metals & Mining,
and Chemicals as systematically missed sectors.

This file contains NO proprietary model weights, features, or signal logic.
It operates on any (Date, Ticker, Prediction, Target_Raw) parquet output.
"""

import os
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — update these paths for your environment
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PREDS_PATH = "path/to/your/model_oos_predictions.parquet"
RAW_DATA_PATH = "path/to/your/nifty500_features.parquet"
INDUSTRY_CSV = "path/to/ind_nifty500list.csv"
OUTPUT_PATH = "path/to/oracle_vs_system_comparison.parquet"

TOP_K = 25
REBALANCE_PERIODS_PER_YEAR = 252 / 20   # ~12.6 periods per year


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: METRICS UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def compute_cagr(returns: pd.Series, ppy: float = REBALANCE_PERIODS_PER_YEAR) -> float:
    """
    Computes CAGR from a series of period returns.

    Args:
        returns: Series of per-period fractional returns
        ppy:     Periods per year (default: 252/20 ≈ 12.6 for monthly rebalancing)

    Returns:
        float: Annualised compound growth rate
    """
    cum = (1 + returns).prod()
    n_years = len(returns) / ppy
    return float(cum ** (1 / n_years) - 1) if n_years > 0 else 0.0


def compute_sharpe(returns: pd.Series, ppy: float = REBALANCE_PERIODS_PER_YEAR) -> float:
    """
    Computes annualised Sharpe ratio from period returns.

    Args:
        returns: Series of per-period fractional returns
        ppy:     Periods per year

    Returns:
        float: Annualised Sharpe ratio
    """
    std = returns.std()
    if std < 1e-9:
        return 0.0
    return float(np.sqrt(ppy) * returns.mean() / std)


def compute_max_drawdown(returns: pd.Series) -> float:
    """
    Computes maximum drawdown from period returns.

    Args:
        returns: Series of per-period fractional returns

    Returns:
        float: Maximum drawdown (negative number)
    """
    cum = (1 + returns).cumprod()
    return float((cum / cum.cummax() - 1).min())


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: BUILD ORACLE vs SYSTEM COMPARISON
# ──────────────────────────────────────────────────────────────────────────────
def build_comparison(
    sys_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    ind_map: dict,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """
    Builds a per-period comparison table of Oracle vs system portfolios.

    For each rebalance date:
      - Oracle selects top_k stocks by actual forward return (Target_Raw)
      - System selects top_k stocks by model prediction (Meta_Pred)
      - Computes overlap, alpha gap, missed/wrong stock returns

    Args:
        sys_df:  System predictions with columns [Date, Ticker, Meta_Pred, Target_Raw]
        raw_df:  Full universe with columns [Date, Ticker, Target_Raw]
        ind_map: Dict mapping Ticker -> Industry
        top_k:   Portfolio size

    Returns:
        pd.DataFrame with one row per rebalance period
    """
    rebalance_dates = sorted(sys_df["Date"].unique())
    print(f"Comparing Oracle vs System across {len(rebalance_dates)} periods...")

    records = []
    for dt in rebalance_dates:
        raw_day = raw_df[raw_df["Date"] == dt]
        sys_day = sys_df[sys_df["Date"] == dt]

        if len(raw_day) < top_k or len(sys_day) < top_k:
            continue

        oracle_top = raw_day.nlargest(top_k, "Target_Raw")
        oracle_stocks = set(oracle_top["Ticker"])
        oracle_ret = oracle_top["Target_Raw"].mean()

        system_stocks = set(sys_day.nlargest(top_k, "Meta_Pred")["Ticker"])
        system_ret = raw_day[raw_day["Ticker"].isin(system_stocks)]["Target_Raw"].mean()

        overlap = len(oracle_stocks & system_stocks)
        missed = oracle_stocks - system_stocks   # Oracle picks we didn't buy
        wrong = system_stocks - oracle_stocks    # Stocks we bought that weren't Oracle

        missed_ret = raw_day[raw_day["Ticker"].isin(missed)]["Target_Raw"].mean() if missed else np.nan
        wrong_ret  = raw_day[raw_day["Ticker"].isin(wrong)]["Target_Raw"].mean()  if wrong  else np.nan

        ts = pd.Timestamp(dt)
        records.append({
            "Date":          ts,
            "Year":          ts.year,
            "Oracle_Ret":    oracle_ret,
            "System_Ret":    system_ret,
            "Alpha_Gap":     oracle_ret - system_ret,
            "Overlap":       overlap,
            "Overlap_Pct":   overlap / top_k,
            "Missed_Ret":    missed_ret,
            "Wrong_Ret":     wrong_ret,
            "Missed_Tickers": sorted(missed),
            "Wrong_Tickers":  sorted(wrong),
        })

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: PRINT REPORTS
# ──────────────────────────────────────────────────────────────────────────────
def print_overall_summary(comp_df: pd.DataFrame) -> None:
    """Prints the high-level capture ratio and overlap statistics."""
    oracle_cagr = compute_cagr(comp_df["Oracle_Ret"])
    system_cagr = compute_cagr(comp_df["System_Ret"])
    capture     = system_cagr / oracle_cagr if oracle_cagr != 0 else np.nan

    print(f"\n{'='*55}")
    print("ORACLE vs SYSTEM — OVERALL COMPARISON")
    print(f"{'='*55}")
    print(f"Oracle CAGR (no costs):    {oracle_cagr:>10.2%}")
    print(f"System CAGR (no costs):    {system_cagr:>10.2%}")
    print(f"Capture Ratio:             {capture:>10.2%}")
    print(f"Avg Overlap per period:    {comp_df['Overlap'].mean():>7.1f} / {TOP_K} stocks "
          f"({comp_df['Overlap_Pct'].mean():.1%})")
    print(f"Avg Alpha Gap per period:  {comp_df['Alpha_Gap'].mean():>10.2%}")
    print(f"Avg Missed stock return:   {comp_df['Missed_Ret'].mean():>10.2%}")
    print(f"Avg Wrong pick return:     {comp_df['Wrong_Ret'].mean():>10.2%}")


def print_per_year_breakdown(comp_df: pd.DataFrame) -> None:
    """Prints year-by-year Oracle vs System comparison."""
    ppy = REBALANCE_PERIODS_PER_YEAR
    print(f"\n{'Year':<6} | {'Oracle':>8} | {'System':>8} | {'Gap':>7} | "
          f"{'Overlap':>8} | {'Missed Ret':>11} | {'Wrong Ret':>10}")
    print("-" * 72)
    for year, g in comp_df.groupby("Year"):
        o_cagr = compute_cagr(g["Oracle_Ret"])
        s_cagr = compute_cagr(g["System_Ret"])
        ov     = g["Overlap"].mean()
        mr     = g["Missed_Ret"].mean()
        wr     = g["Wrong_Ret"].mean()
        print(f"{year:<6} | {o_cagr:>8.1%} | {s_cagr:>8.1%} | {o_cagr-s_cagr:>7.1%} | "
              f"{ov:>7.1f}/25 | {mr:>11.2%} | {wr:>10.2%}")
    print("-" * 72)


def print_sector_analysis(comp_df: pd.DataFrame, ind_map: dict) -> None:
    """Prints sector enrichment analysis of missed Oracle stocks."""
    all_missed = []
    for _, row in comp_df.iterrows():
        for t in row["Missed_Tickers"]:
            all_missed.append({"Ticker": t, "Year": row["Year"]})

    missed_df = pd.DataFrame(all_missed)
    if missed_df.empty:
        return

    missed_df["Industry"] = missed_df["Ticker"].map(ind_map).fillna("Unknown")

    universe_ind = pd.Series(list(ind_map.values())).value_counts()
    miss_by_ind  = missed_df["Industry"].value_counts().head(10)

    print("\n\nINDUSTRY BREAKDOWN — WHERE ARE WE MISSING ORACLE PICKS?")
    print(f"\n{'Sector':<40} {'Misses':>7}  {'Enrichment':>12}")
    print("-" * 65)
    for ind, cnt in miss_by_ind.items():
        universe_cnt = universe_ind.get(ind, 1)
        enrichment   = (cnt / len(missed_df)) / (universe_cnt / len(ind_map))
        print(f"{ind:<40} {cnt:>7}  {enrichment:>11.2f}×")


def print_hardest_periods(comp_df: pd.DataFrame) -> None:
    """Prints periods with the lowest overlap and largest alpha gaps."""
    print("\n\nHARDEST PERIODS (lowest overlap):")
    hard = comp_df.nsmallest(10, "Overlap")[
        ["Date", "Year", "Overlap", "Alpha_Gap", "Oracle_Ret", "System_Ret"]
    ]
    print(hard.to_string(index=False))

    print("\n\nBIGGEST ALPHA GAPS (most return left on the table):")
    gap = comp_df.nlargest(10, "Alpha_Gap")[
        ["Date", "Year", "Overlap", "Alpha_Gap", "Oracle_Ret", "System_Ret", "Missed_Ret"]
    ]
    print(gap.to_string(index=False))


def print_return_distribution(comp_df: pd.DataFrame) -> None:
    """Prints return distribution statistics for missed vs wrong picks."""
    print("\n\nRETURN DISTRIBUTION ANALYSIS:")
    for label, col in [("Missed picks (Oracle stocks we didn't buy)", "Missed_Ret"),
                        ("Wrong picks (stocks we bought that weren't Oracle)", "Wrong_Ret")]:
        s = comp_df[col].dropna()
        print(f"\n{label}:")
        print(f"  Mean:   {s.mean():.2%}")
        print(f"  Median: {s.median():.2%}")
        print(f"  Std:    {s.std():.2%}")
        if col == "Missed_Ret":
            print(f"  Periods where missed return > 10%: {(s > 0.10).mean():.1%}")
        else:
            print(f"  Periods where wrong picks lost money: {(s < 0).mean():.1%}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  ORACLE vs SYSTEM COMPARISON")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────────
    demo_mode = not os.path.exists(SYSTEM_PREDS_PATH)

    if demo_mode:
        print("\n[DEMO MODE] Prediction file not found — generating synthetic data.\n")
        np.random.seed(42)
        dates = pd.date_range("2015-01-01", periods=100, freq="20B")
        tickers = [f"STOCK_{i:03d}" for i in range(100)]
        rows = []
        for dt in dates:
            for t in tickers:
                rows.append({
                    "Date": dt, "Ticker": t,
                    "Meta_Pred": np.random.randn(),
                    "Target_Raw": np.random.normal(0.03, 0.12),
                })
        sys_df = pd.DataFrame(rows)
        raw_df = sys_df[["Date", "Ticker", "Target_Raw"]].copy()
        ind_map = {t: np.random.choice(["Financials","IT","Energy","Metals","Consumer"]) for t in tickers}
    else:
        print(f"Loading system predictions from: {SYSTEM_PREDS_PATH}")
        sys_df = pd.read_parquet(SYSTEM_PREDS_PATH)

        # Normalise column names
        if "Pred" in sys_df.columns and "Meta_Pred" not in sys_df.columns:
            sys_df = sys_df.rename(columns={"Pred": "Meta_Pred"})
        if "Target" in sys_df.columns and "Target_Raw" not in sys_df.columns:
            sys_df = sys_df.rename(columns={"Target": "Target_Raw"})

        print(f"Loading raw universe from: {RAW_DATA_PATH}")
        raw_df = pd.read_parquet(RAW_DATA_PATH)
        raw_col = "Target_Raw" if "Target_Raw" in raw_df.columns else "Target"
        raw_df = raw_df[["Date", "Ticker", raw_col]].rename(columns={raw_col: "Target_Raw"})

        ind_map = {}
        if os.path.exists(INDUSTRY_CSV):
            ind_df  = pd.read_csv(INDUSTRY_CSV)
            ind_map = ind_df.set_index("Symbol")["Industry"].to_dict()

    sys_df["Date"] = pd.to_datetime(sys_df["Date"])
    raw_df["Date"] = pd.to_datetime(raw_df["Date"])

    # ── Build comparison ───────────────────────────────────────────────────────
    comp_df = build_comparison(sys_df, raw_df, ind_map)

    if comp_df.empty:
        print("No overlapping periods found between system predictions and raw data.")
        return

    # ── Print all reports ──────────────────────────────────────────────────────
    print_overall_summary(comp_df)
    print_per_year_breakdown(comp_df)
    print_sector_analysis(comp_df, ind_map)
    print_hardest_periods(comp_df)
    print_return_distribution(comp_df)

    # ── Save output ────────────────────────────────────────────────────────────
    save_df = comp_df.copy()
    save_df["Missed_Tickers"] = save_df["Missed_Tickers"].apply(
        lambda x: ",".join(x) if isinstance(x, list) else ""
    )
    save_df["Wrong_Tickers"] = save_df["Wrong_Tickers"].apply(
        lambda x: ",".join(x) if isinstance(x, list) else ""
    )
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    save_df.to_parquet(OUTPUT_PATH)
    print(f"\nSaved comparison table to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
