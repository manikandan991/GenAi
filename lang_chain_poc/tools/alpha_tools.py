import os, requests

AV_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")

def _av_call(function: str, symbol: str):
    url = "https://www.alphavantage.co/query"
    r = requests.get(url, params={"function": function, "symbol": symbol, "apikey": AV_KEY}, timeout=30)
    r.raise_for_status()
    return r.json()

def _finnhub_financials(symbol: str):
    # Simple fallback: Finnhub "financials reported" (quarterly/annual)
    # NOTE: adjust to the exact endpoint you prefer; keep it minimal for POC
    if not FINNHUB_KEY:
        return {}
    base = "https://finnhub.io/api/v1"
    headers = {}
    params = {"symbol": symbol, "token": FINNHUB_KEY}
    # Example: basic metrics endpoint (broad coverage)
    m = requests.get(f"{base}/stock/metric", params={**params, "metric": "all"}, timeout=30).json()
    # You can also pull /financials-reported and compute metrics from statements
    return {"metric": m}

def alpha_fetch_statements(symbol: str):
    # 1) Try AV fundamentals first
    annual_is = _av_call("INCOME_STATEMENT", symbol)
    annual_bs = _av_call("BALANCE_SHEET", symbol)
    annual_cf = _av_call("CASH_FLOW", symbol)

    def _has_data(obj): 
        return isinstance(obj, dict) and any(k for k in obj.keys() if "annual" in k.lower() or "quarterly" in k.lower())

    if _has_data(annual_is) or _has_data(annual_bs) or _has_data(annual_cf):
        return {
            "provider": "alphavantage",
            "symbol": symbol,
            "annual_is": annual_is,
            "annual_bs": annual_bs,
            "annual_cf": annual_cf,
        }

    # 2) Fallback to Finnhub for NSE/BSE etc.
    fb = _finnhub_financials(symbol)
    return {
        "provider": "finnhub" if fb else "none",
        "symbol": symbol,
        "fallback": fb,
    }

def analyze_alpha_fundamentals(symbol: str):
    data = alpha_fetch_statements(symbol)

    # If Finnhub fallback, compute basic signals from what you fetched (metric section),
    # or return a clear message asking to supply a symbol with coverage.
    if data.get("provider") == "finnhub":
        metric = data.get("fallback", {}).get("metric", {})
        # Example: pick a few robust fields if present
        ratios = (metric.get("metric") or {})
        de = ratios.get("totalDebt/totalEquityAnnual", None) or ratios.get("debtToEquity", None)
        icr = ratios.get("interestCoverage", None)
        # Compute a coarse verdict
        verdict = "OK" if (de is not None and de < 0.5) and (icr is not None and icr >= 4) else "Mixed/Weak (incomplete)"
        return {
            "symbol": symbol,
            "provider": "finnhub",
            "metrics": {
                "de_ratio": de,
                "interest_coverage": icr,
                "revenue_cagr_5y": None,   # add if you derive from reports
                "pat_cagr_5y": None,
                "cash_conversion": None,
            },
            "rating": "OK" if verdict == "OK" else "Weak",
            "verdict": verdict,
            "raw": data.get("fallback"),
        }

    if data.get("provider") == "alphavantage":
        # Your existing AV parsing -> compute CAGR, D/E, ICR, CFO/PAT etc.
        # (re-use what you already wrote)
        return _compute_metrics_from_av_payloads(data)

    # Nothing usable
    return {
        "symbol": symbol,
        "metrics": {k: None for k in ["revenue_cagr_5y","pat_cagr_5y","de_ratio","interest_coverage","cash_conversion"]},
        "rating": "Weak",
        "verdict": "No fundamentals available from Alpha Vantage and no fallback provider configured.",
        "raw": {},
    }
