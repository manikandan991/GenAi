from typing import Dict, Any, List
from datetime import datetime
import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def fetch_news_sentiment(ticker: str, limit: int = 10) -> Dict[str, Any]:
    t = yf.Ticker(ticker)
    news = t.news or []
    rows = []
    for n in news[:limit]:
        title = n.get("title", "")
        url = n.get("link", "")
        pub_ts = n.get("providerPublishTime")
        pub_dt = datetime.utcfromtimestamp(pub_ts) if pub_ts else None
        s = analyzer.polarity_scores(title)
        rows.append({"title": title, "url": url, "published": pub_dt, "compound": s["compound"]})
    if not rows:
        return {"positive": [], "negative": [], "all": []}
    df = pd.DataFrame(rows).sort_values("published", ascending=False)
    pos = df[df["compound"] > 0.25].head(5).to_dict(orient="records")
    neg = df[df["compound"] < -0.25].head(5).to_dict(orient="records")
    return {"positive": pos, "negative": neg, "all": df.to_dict(orient="records")}
