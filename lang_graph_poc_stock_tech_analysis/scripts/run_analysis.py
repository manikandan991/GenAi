import os, argparse, json
os.makedirs("./artifacts/charts", exist_ok=True)
os.makedirs("./artifacts/reports", exist_ok=True)

from src.rag.graph import build_graph, AgentState

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g., AAPL or TCS.NS")
    args = parser.parse_args()
    graph = build_graph(kb_dir="./src/rag/kb", chroma_dir="./artifacts/chroma")
    init = AgentState(query=args.ticker, ticker="")
    out = graph.invoke(init)
    payload = {
        "ticker": out.ticker,
        "analysis": out.draft,
        "chart_image": out.ta.chart_path if out.ta else None,
        "signals": out.ta.signals if out.ta else [],
        "trend": out.ta.trend if out.ta else "unknown",
    }
    print(json.dumps(payload, default=str, indent=2))

if __name__ == "__main__":
    main()
