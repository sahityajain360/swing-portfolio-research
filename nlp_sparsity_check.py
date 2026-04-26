"""
nlp_sparsity_check.py
=====================
NLP Micro-Sentiment Sparsity Audit

This script quantifies the coverage of idiosyncratic (ticker-level) news
sentiment in a financial NLP dataset. It was used to explain why
Micro-Sentiment features failed to add alpha in the swing portfolio system,
despite Macro-Sentiment (aggregated economy-level sentiment) ranking 9th
out of 130+ features by importance.

Key findings from running this on 12 years of Indian financial news:
  - 99.59% of stock-days had ZERO idiosyncratic news coverage
  - 73.9% of Nifty 500 tickers had zero articles across the full 12-year period
  - Median articles per stock: 0 (most stocks are completely invisible)
  - Top coverage: "OIL" with 2,686 articles — likely inflated by false positives
    from the common English word appearing in non-financial contexts
  - The ticker "IDEA" similarly inflated by common language usage

These findings have broader implications for NLP alpha research in emerging
markets: open-source newspaper archives are inadequate for idiosyncratic
stock-level signals on mid/small-cap Indian equities. Premium data sources
(Bloomberg, Reuters, earnings call transcripts) would be required.

This file contains NO proprietary model logic or trading signals.
It is a pure data characterisation and audit tool.
"""

import pandas as pd
import numpy as np
import os


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
NLP_DATA_PATH = "path/to/nlp_dual_sentiment_features.parquet"

# Expected columns in the NLP parquet
COL_TICKER    = "Ticker"
COL_DATE      = "Date"
COL_MICRO_VOL = "Micro_News_Vol"      # Article count per (Ticker, Date)
COL_MACRO_VOL = "Macro_News_Vol"      # Aggregate economy-level article count per Date
COL_MICRO_SENT = "Micro_Sent_7d"     # 7-day rolling micro-sentiment score
COL_MACRO_SENT = "Macro_Sent_7d"     # 7-day rolling macro-sentiment score


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD OR GENERATE DATA
# ──────────────────────────────────────────────────────────────────────────────
def load_nlp_data(path: str) -> pd.DataFrame:
    """
    Loads the NLP sentiment feature parquet.
    Falls back to synthetic data if path not found (for demo/testing).

    Args:
        path: Filesystem path to the NLP parquet file

    Returns:
        pd.DataFrame with sentiment features
    """
    if not os.path.exists(path):
        print(f"[DEMO MODE] File not found at: {path}")
        print("Generating synthetic NLP data to demonstrate analysis structure...\n")
        return _generate_synthetic_nlp_data()

    print(f"Loading NLP data from: {path}")
    df = pd.read_parquet(path)
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    return df


def _generate_synthetic_nlp_data(
    n_tickers: int = 490,
    n_dates: int = 2760,   # ~11 years of trading days
) -> pd.DataFrame:
    """
    Generates synthetic NLP data that mirrors the real sparsity structure.

    The synthetic data is calibrated to reproduce the key finding:
    ~99.5% of stock-days have zero idiosyncratic news coverage.

    Args:
        n_tickers: Number of stocks in the universe
        n_dates:   Number of trading days in the dataset

    Returns:
        pd.DataFrame with realistic sparsity structure
    """
    np.random.seed(42)

    tickers = [f"STOCK_{i:03d}" for i in range(n_tickers)]
    dates   = pd.date_range("2012-01-01", periods=n_dates, freq="B")

    rows = []
    for date in dates:
        for ticker in tickers:
            # 99.5% sparsity — only 0.5% of stock-days have any news
            has_news = np.random.random() < 0.005

            # Macro sentiment has much better coverage (~70% of dates)
            macro_vol  = np.random.poisson(3) if np.random.random() < 0.70 else 0
            macro_sent = np.random.uniform(-0.3, 0.3) if macro_vol > 0 else 0.0

            micro_vol  = np.random.poisson(1) if has_news else 0
            micro_sent = np.random.uniform(-0.5, 0.5) if has_news else 0.0

            rows.append({
                COL_DATE:      date,
                COL_TICKER:    ticker,
                COL_MICRO_VOL: micro_vol,
                COL_MICRO_SENT: micro_sent,
                COL_MACRO_VOL: macro_vol,
                COL_MACRO_SENT: macro_sent,
            })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: TICKER-LEVEL COVERAGE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
def analyze_ticker_coverage(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Measures total article coverage per ticker across the full dataset.

    Args:
        df:    NLP DataFrame with Ticker and Micro_News_Vol columns
        top_n: How many tickers to show at each extreme

    Returns:
        pd.Series of total articles per ticker, sorted descending
    """
    ticker_vol = df.groupby(COL_TICKER)[COL_MICRO_VOL].sum().sort_values(ascending=False)

    total_tickers  = len(ticker_vol)
    with_news      = (ticker_vol > 0).sum()
    without_news   = (ticker_vol == 0).sum()

    print(f"\n{'='*55}")
    print("TICKER-LEVEL NEWS COVERAGE")
    print(f"{'='*55}")
    print(f"Total tickers in universe:        {total_tickers:>6}")
    print(f"Tickers with ≥1 article:          {with_news:>6}  ({with_news/total_tickers:.1%})")
    print(f"Tickers with ZERO articles:       {without_news:>6}  ({without_news/total_tickers:.1%})")
    print(f"\nAverage articles per ticker:      {ticker_vol.mean():>8.1f}")
    print(f"Median  articles per ticker:      {ticker_vol.median():>8.1f}")
    print(f"(Median = 0 means majority have zero coverage)")

    print(f"\n--- TOP {top_n} MOST COVERED TICKERS ---")
    print(ticker_vol.head(top_n).to_string())

    non_zero = ticker_vol[ticker_vol > 0]
    if len(non_zero) >= top_n:
        print(f"\n--- BOTTOM {top_n} LEAST COVERED TICKERS (Excluding Zeros) ---")
        print(non_zero.tail(top_n).to_string())

    return ticker_vol


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: GRID SPARSITY
# ──────────────────────────────────────────────────────────────────────────────
def analyze_grid_sparsity(df: pd.DataFrame) -> dict:
    """
    Measures sparsity of the stock-day grid.

    Sparsity = fraction of (Ticker × Date) cells with zero news.
    This is the key metric demonstrating why micro-sentiment failed.

    Args:
        df: NLP DataFrame

    Returns:
        dict with total_rows, rows_with_news, sparsity
    """
    total_rows    = len(df)
    rows_with_news = (df[COL_MICRO_VOL] > 0).sum()
    sparsity      = 1 - (rows_with_news / total_rows)

    print(f"\n{'='*55}")
    print("GRID SPARSITY (Stock-Day Level)")
    print(f"{'='*55}")
    print(f"Total stock-days in dataset:      {total_rows:>12,}")
    print(f"Stock-days WITH news:             {rows_with_news:>12,}")
    print(f"Stock-days with ZERO news:        {total_rows-rows_with_news:>12,}")
    print(f"\nDataset sparsity:                 {sparsity:>11.2%}")
    print(f"\nInterpretation: {sparsity:.1%} of the dataset is structurally zero.")
    print("This means any ML model trained on Micro_Sent features is essentially")
    print("learning from noise 99%+ of the time — the signal is too sparse to generalise.")

    return {
        "total_rows":     total_rows,
        "rows_with_news": rows_with_news,
        "sparsity":       sparsity,
    }


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: MACRO vs MICRO COMPARISON
# ──────────────────────────────────────────────────────────────────────────────
def compare_macro_vs_micro(df: pd.DataFrame) -> None:
    """
    Contrasts the coverage and density of Macro vs Micro sentiment.

    Macro sentiment aggregates ALL articles per date (economy-level).
    Micro sentiment is ticker-specific. This comparison explains why
    Macro_Sent_Shock ranked 9th in feature importance while all
    Micro_Sent features failed to appear in the top 15.

    Args:
        df: NLP DataFrame
    """
    print(f"\n{'='*55}")
    print("MACRO vs MICRO SENTIMENT COVERAGE COMPARISON")
    print(f"{'='*55}")

    macro_coverage = (df[COL_MACRO_VOL] > 0).mean() if COL_MACRO_VOL in df.columns else None
    micro_coverage = (df[COL_MICRO_VOL] > 0).mean()

    if macro_coverage is not None:
        print(f"Macro sentiment coverage:  {macro_coverage:>8.1%} of stock-days")
    print(f"Micro sentiment coverage:  {micro_coverage:>8.1%} of stock-days")

    if macro_coverage is not None:
        ratio = macro_coverage / micro_coverage if micro_coverage > 0 else float('inf')
        print(f"\nMacro is {ratio:.0f}× better covered than Micro sentiment.")

    print("\nWhy Macro works but Micro doesn't:")
    print("  Macro: aggregates ALL economy-wide news → dense, consistent signal")
    print("  Micro: requires articles mentioning a specific ticker → 99.6% empty")
    print("  Even with good articles, regex ticker-matching introduces false positives")
    print("  e.g. 'OIL' triggers on any article containing the word 'oil'")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5: TEMPORAL COVERAGE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
def analyze_temporal_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Measures how news coverage evolved over the years.

    Useful for detecting structural breaks — e.g., if digital news archives
    only became comprehensive after a certain year.

    Args:
        df: NLP DataFrame with Date column

    Returns:
        pd.DataFrame with year-level coverage statistics
    """
    df = df.copy()
    df["Year"] = pd.to_datetime(df[COL_DATE]).dt.year

    annual = df.groupby("Year").agg(
        total_rows    = (COL_MICRO_VOL, "count"),
        rows_with_news= (COL_MICRO_VOL, lambda x: (x > 0).sum()),
        total_articles= (COL_MICRO_VOL, "sum"),
    ).reset_index()

    annual["coverage_pct"] = annual["rows_with_news"] / annual["total_rows"]

    print(f"\n{'='*55}")
    print("TEMPORAL COVERAGE BY YEAR")
    print(f"{'='*55}")
    print(f"{'Year':<6} {'Stock-Days':>11} {'With News':>10} {'Coverage':>10} {'Articles':>10}")
    print("-" * 55)
    for _, row in annual.iterrows():
        print(f"{int(row['Year']):<6} {int(row['total_rows']):>11,} "
              f"{int(row['rows_with_news']):>10,} "
              f"{row['coverage_pct']:>10.2%} "
              f"{int(row['total_articles']):>10,}")

    return annual


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  NLP MICRO-SENTIMENT SPARSITY AUDIT")
    print("  Why idiosyncratic NLP alpha failed in Indian equities")
    print("=" * 60)

    df = load_nlp_data(NLP_DATA_PATH)

    print(f"\nDataset shape: {df.shape}")
    print(f"Columns:       {list(df.columns)}")
    if COL_DATE in df.columns:
        df[COL_DATE] = pd.to_datetime(df[COL_DATE])
        date_range = f"{df[COL_DATE].min().date()} to {df[COL_DATE].max().date()}"
        print(f"Date range:    {date_range}")

    ticker_coverage = analyze_ticker_coverage(df)
    sparsity_stats  = analyze_grid_sparsity(df)
    compare_macro_vs_micro(df)
    temporal_stats  = analyze_temporal_coverage(df)

    print(f"\n{'='*60}")
    print("SUMMARY OF KEY FINDINGS")
    print(f"{'='*60}")
    print(f"1. Grid sparsity:        {sparsity_stats['sparsity']:.2%} zero stock-days")
    print(f"2. Tickers with no news: "
          f"{(ticker_coverage == 0).sum()} / {len(ticker_coverage)} "
          f"({(ticker_coverage == 0).mean():.1%})")
    print(f"3. Median total articles per ticker: {ticker_coverage.median():.0f}")
    print(f"4. Implication: Open-source NLP data is insufficient for")
    print(f"   idiosyncratic alpha on Nifty 500 equities.")
    print(f"   Macro-level aggregated sentiment (Macro_Sent_Shock) works;")
    print(f"   ticker-level micro-sentiment does not.")


if __name__ == "__main__":
    main()
