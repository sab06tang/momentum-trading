import yfinance as yf
import pandas as pd


def load_data(
    tickers: list[str],
    start_date: str,
    end_date: str | None = None,
    ffill_limit: int = 5,
) -> pd.DataFrame:
    """
    Downloads adjusted daily close prices for a cross-asset universe.

    Parameters
    ----------
    tickers     : list of yfinance-compatible ticker symbols
    start_date  : 'YYYY-MM-DD' inclusive start
    end_date    : 'YYYY-MM-DD' exclusive end (yfinance convention); defaults to today
    ffill_limit : max consecutive days to forward-fill (guards against stale prices)

    Returns
    -------
    pd.DataFrame of adjusted close prices, indexed by date, one column per ticker.
    Rows before all tickers have valid data are dropped; a warning is printed if
    this trims more than 0 rows.
    """
    print(f"Downloading {len(tickers)} tickers from {start_date} "
          f"to {end_date or 'today'}...")

    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,   # split- and dividend-adjusted closes
        progress=False,
    )

    prices = raw["Close"]

    # normalize single-ticker download from Series to DataFrame
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    if prices.empty:
        raise ValueError(
            f"No price data returned for {tickers}. "
            "Verify ticker symbols and date range."
        )

    prices = prices.dropna(how="all")        # drop fully-empty rows (market holidays)
    prices = prices.ffill(limit=ffill_limit) # bounded forward-fill for gaps

    n_before = len(prices)
    prices = prices.dropna(how="any")        # trim leading rows (unequal start dates)
    trimmed = n_before - len(prices)
    if trimmed > 0:
        print(f"  [WARN] Dropped {trimmed} leading rows due to unequal start dates. "
              f"Effective start: {prices.index[0].date()}")

    print(f"  Loaded {len(prices)} trading days × {len(prices.columns)} assets. "
          f"({prices.index[0].date()} → {prices.index[-1].date()})")
    return prices