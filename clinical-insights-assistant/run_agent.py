# Databricks notebook source
# %pip install -r /Workspace/Users/manikandan_nagarasan@epam.com/clinical-insights-assistant/requirements.txt
# dbutils.library.restartPython()

# COMMAND ----------

# %restart_python

# COMMAND ----------

from dotenv import load_dotenv
load_dotenv("/Workspace/Users/manikandan_nagarasan@epam.com/clinical-insights-assistant/.env")


# COMMAND ----------

# %run /Workspace/Users/manikandan_nagarasan@epam.com/clinical-insights-assistant/data/generate_synth_data

# COMMAND ----------

# === Clinical Insights Assistant — ipywidgets UI with matplotlib flow ===
# Paste this in a Databricks notebook cell and run.

import os, time
import ipywidgets as W
from IPython.display import display, Markdown, clear_output

# --- your project imports ---
from src.agent.agent_core import run_agent
from src.data_loader import load_data
from src.cohort_analysis import cohort_summary

# --- matplotlib for flow diagram (no JS; works in Databricks) ---
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

# ---------------- Defaults ----------------
PROJECT_DIR = "/Workspace/Users/manikandan_nagarasan@epam.com/clinical-insights-assistant"
DEFAULT_DATA = f"{PROJECT_DIR}/data/clinical_trial_data.csv"
os.environ.setdefault("DATA_PATH", DEFAULT_DATA)

# ---------------- Widgets -----------------
title = W.HTML("<h2>🧪 Clinical Insights Assistant (GenAI POC)</h2>")

data_path = W.Text(
    value=os.getenv("DATA_PATH", DEFAULT_DATA),
    description="DATA_PATH",
    layout=W.Layout(width="100%")
)
reload_btn = W.Button(description="Reload Data", button_style="info")
tip = W.HTML("<span style='color:#666'>Tip: set OPENAI_API_KEY via cluster env or notebook env.</span>")

dataset_hdr = W.HTML("<h3>Dataset Preview</h3>")
cohort_hdr  = W.HTML("<h3>Cohort Snapshot</h3>")
query_hdr   = W.HTML("<hr/><h3>Ask the Assistant</h3>")
query = W.Text(
    value="Summarize doctor feedback and flag issues for cohort A last 7 days",
    description="Your request",
    layout=W.Layout(width="100%")
)
run_btn = W.Button(description="Run", button_style="primary")

status     = W.HTML("<b>Status:</b> idle")
table_out  = W.Output()
cohort_out = W.Output()
result_out = W.Output()
flow_out   = W.Output()

# --------------- Flow diagram data ----------------
# (key, label, x, y)
NODES = [
    ("entry",   "Entry",           0.00, 0.50),
    ("router",  "Router",          0.20, 0.50),
    ("issues",  "Issues",          0.45, 0.80),
    ("cohorts", "Cohorts",         0.45, 0.55),
    ("simulate","Simulate",        0.45, 0.30),
    ("genai",   "GenAI Summary",   0.45, 0.05),
    ("end",     "End",             0.80, 0.50),
]
EDGES = [
    ("entry","router"),
    ("router","issues"),
    ("router","cohorts"),
    ("router","simulate"),
    ("router","genai"),
    ("issues","end"),
    ("cohorts","end"),
    ("simulate","end"),
    ("genai","end"),
]
pos_by_key  = {k:(x,y) for k,_,x,y in NODES}

# Colors
COLOR_IDLE_FILL   = "#CFD8DC"  # grey
COLOR_IDLE_EDGE   = "#607D8B"
COLOR_ACTIVE_FILL = "#FFD54F"  # amber
COLOR_ACTIVE_EDGE = "#FF6F00"
COLOR_DONE_FILL   = "#A5D6A7"  # green
COLOR_DONE_EDGE   = "#2E7D32"

def _build_fig_mpl(active=set(), done=set()):
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")

    # Edges
    for src, dst in EDGES:
        x0, y0 = pos_by_key[src]
        x1, y1 = pos_by_key[dst]
        arrow = FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>",
            mutation_scale=14,
            lw=1.5,
            color="#90A4AE"
        )
        ax.add_patch(arrow)

    # Nodes
    for k, label, x, y in NODES:
        if k in active:
            fc, ec, r = COLOR_ACTIVE_FILL, COLOR_ACTIVE_EDGE, 0.035
        elif k in done:
            fc, ec, r = COLOR_DONE_FILL, COLOR_DONE_EDGE, 0.03
        else:
            fc, ec, r = COLOR_IDLE_FILL, COLOR_IDLE_EDGE, 0.028
        circ = Circle((x, y), r, facecolor=fc, edgecolor=ec, linewidth=2)
        ax.add_patch(circ)
        ax.text(x, y + 0.055, label, ha="center", va="center", fontsize=10)

    fig.tight_layout()
    return fig

def show_flow(active=set(), done=set()):
    with flow_out:
        clear_output(wait=True)
        fig = _build_fig_mpl(active, done)
        display(fig)
        plt.close(fig)

def step_flow(sequence, delay=0.45):
    done = set()
    for k in sequence:
        show_flow(active={k}, done=done)
        time.sleep(delay)
        done.add(k)
    show_flow(active=set(), done=done)

# --------------- Data handlers --------------------
def reload_data(_=None):
    with table_out:
        clear_output()
        try:
            df = load_data(data_path.value)
            display(df.head(20))
            status.value = "<b>Status:</b> data loaded ✅"
        except Exception as e:
            status.value = f"<b>Status:</b> ❌ Failed to load data from {data_path.value}<br/>{e}"
    with cohort_out:
        clear_output()
        try:
            df = load_data(data_path.value)
            # display(cohort_summary(df))
        except Exception as e:
            print("Cohort summary error:", e)
    show_flow(active=set(), done={"entry"})

def _tool_key_from_result(res: dict):
    t = (res or {}).get("selected_tool") or ""
    t = t.lower()
    if "issue" in t:                              return "issues"
    if "cohort" in t:                             return "cohorts"
    if "simulate" in t or "scenario" in t:        return "simulate"
    return "genai"  # default

def on_run(_):
    with result_out:
        clear_output()
        status.value = "<b>Status:</b> analyzing…"
        step_flow(["entry","router"], delay=0.35)
        try:
            res = run_agent(query.value)  # uses your existing run_agent (with checkpointer config)
            tool_key = _tool_key_from_result(res)
            status.value = f"<b>Status:</b> executing {tool_key}…"
            step_flow(["entry","router", tool_key, "end"], delay=0.3)

            tool = res.get("selected_tool")
            txt  = res.get("result_text") or ""
            summary = res.get("summary")

            display(Markdown(f"**Selected Tool:** `{tool}`"))
            display(Markdown("**Result Text:**"))
            print(txt)
            if summary:
                display(Markdown("**Regulatory-style Summary:**"))
                display(Markdown(summary))
            status.value = "<b>Status:</b> done ✅"
        except Exception as e:
            show_flow(active=set(), done={"entry","router"})
            status.value = f"<b>Status:</b> ❌ {type(e).__name__}: {e}"

reload_btn.on_click(reload_data)
run_btn.on_click(on_run)

# ---------------- Layout -----------------
controls = W.VBox([data_path, reload_btn, tip])
flow_hdr = W.HTML("<h3>Flow</h3>")
preview  = W.VBox([dataset_hdr, table_out]) #, cohort_hdr, cohort_out])
ask      = W.VBox([query_hdr, query, run_btn, result_out])

ui = W.VBox([title, controls, preview, ask, status, flow_hdr, flow_out,])
display(ui)

# ---------------- Init -------------------
show_flow(active=set(), done={"entry"})
reload_data()
