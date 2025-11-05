from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import ta

@dataclass
class TAResult:
    ticker: str
    df_tail: List[Dict[str, Any]]
    sma50: float
    sma200: float
    rsi: float
    macd: float
    macd_signal: float
    bb_width: float
    trend: str
    signals: List[str]
    chart_path: str

def fetch_ohlc(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    t = yf.Ticker(ticker)
    df = t.history(period=period, interval=interval, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"No data for {ticker}")
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SMA50"] = ta.trend.sma_indicator(out["Close"], window=50)
    out["SMA200"] = ta.trend.sma_indicator(out["Close"], window=200)
    out["RSI"] = ta.momentum.rsi(out["Close"], window=14)
    out["MACD"] = ta.trend.macd(out["Close"], window_slow=26, window_fast=12)
    out["MACD_SIGNAL"] = ta.trend.macd_signal(out["Close"], window_slow=26, window_fast=12, window_sign=9)
    bb_high = ta.volatility.bollinger_hband(out["Close"], window=20, window_dev=2)
    bb_low = ta.volatility.bollinger_lband(out["Close"], window=20, window_dev=2)
    out["BB_WIDTH"] = (bb_high - bb_low) / out["Close"]
    return out

def plot_chart(df: pd.DataFrame, ticker: str, charts_dir: str = "./artifacts/charts") -> str:
    import os
    os.makedirs(charts_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["Close"], label="Close")
    if "SMA50" in df:
        ax.plot(df.index, df["SMA50"], label="SMA50")
    if "SMA200" in df:
        ax.plot(df.index, df["SMA200"], label="SMA200")
    ax.set_title(f"{ticker} — Close with SMA50/200")
    ax.legend()
    path = os.path.join(charts_dir, f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    return path

def analyze_ta(ticker: str, charts_dir: str = "./artifacts/charts") -> TAResult:
    df = fetch_ohlc(ticker)
    df = compute_indicators(df)
    latest = df.dropna().iloc[-1]
    signals = []
    trend = "bullish" if latest["SMA50"] > latest["SMA200"] else "bearish" if latest["SMA50"] < latest["SMA200"] else "unknown"
    prev = df.dropna().iloc[-2]
    if prev["MACD"] < prev["MACD_SIGNAL"] and latest["MACD"] > latest["MACD_SIGNAL"]:
        signals.append("MACD bullish crossover")
    if prev["MACD"] > prev["MACD_SIGNAL"] and latest["MACD"] < latest["MACD_SIGNAL"]:
        signals.append("MACD bearish crossover")
    if latest["RSI"] < 35: signals.append("RSI near oversold")
    if latest["RSI"] > 65: signals.append("RSI near overbought")
    chart_path = plot_chart(df, ticker, charts_dir=charts_dir)
    return TAResult(
        ticker=ticker,
        df_tail=df.tail(5).reset_index().to_dict(orient="records"),
        sma50=float(latest["SMA50"]), sma200=float(latest["SMA200"]),
        rsi=float(latest["RSI"]), macd=float(latest["MACD"]), macd_signal=float(latest["MACD_SIGNAL"]),
        bb_width=float(latest["BB_WIDTH"]), trend=trend, signals=signals, chart_path=chart_path
    )
