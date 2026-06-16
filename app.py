
import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from src.fetcher import fetch_articles
from src.embedder import embed_and_store
from src.analyzer import cluster_articles, analyze_with_llm
from src.agent import chat as agent_chat


EXAMPLES = [
    "OpenAI",
    "Indian economy",
    "Climate policy",
    "Fed rate cuts",
    "India-Pakistan",
    "Nvidia",
    "US elections",
    "Bitcoin",
    "Gaza conflict",
    "AI regulation",
]


THEME_JS = r"""
() => {
    const b = document.body;
    const btn = document.querySelector('#tp-theme-fixed button');
    if (!btn) return;

    if (b.classList.contains('tp-dark')) {
        b.classList.remove('tp-dark');
        btn.textContent = 'Dark mode';
    } else {
        b.classList.add('tp-dark');
        btn.textContent = 'Light mode';
    }
}
"""

CSS = r"""
body{
    --bg:#F8FAFC;
    --surf:#FFFFFF;
    --surf2:#F1F5F9;
    --bdr:#E2E8F0;
    --txt:#0F172A;
}

input[type=text],
input[type=number],
textarea{
    background:var(--surf) !important;
    color:var(--txt) !important;
    border:1.5px solid var(--bdr) !important;
    border-radius:8px !important;
    font-size:14px !important;
}

footer{
    display:none !important;
}
"""

with gr.Blocks(
    title="TrendPulse",
    css=CSS,
    theme=gr.themes.Base()
) as demo:

    gr.Markdown("# TrendPulse")

    with gr.Tab("Analyse Topic"):
        topic_input = gr.Textbox(label="Topic")
        analyse_btn = gr.Button("Analyse")

    with gr.Tab("Agent Chat"):
        gr.ChatInterface(fn=agent_chat)

if __name__ == "__main__":
    demo.launch(show_api=False)
