import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from src.fetcher import fetch_articles
from src.embedder import embed_and_store
from src.analyzer import cluster_articles, analyze_with_llm
from src.agent import chat as agent_chat

EXAMPLES = [
    "OpenAI", "Indian economy", "Climate policy",
    "Fed rate cuts", "India-Pakistan", "Nvidia",
    "US elections", "Bitcoin", "Gaza conflict", "AI regulation"
]

def sentiment_color(score):
    if score > 0.1: return "#16A34A"
    if score < -0.1: return "#DC2626"
    return "#6B7280"

def build_summary(analysis, topic):
    meta = analysis["meta"]
    md = f"## {topic}\n\n"
    md += f"**{analysis['total_articles']} articles analysed** &nbsp;·&nbsp; **{meta['overall_sentiment'].title()} sentiment**\n\n"
    md += f"> {meta['dominant_narrative']}\n\n"
    if meta.get("contradictions"):
        md += "**Contradictions detected:**\n"
        for c in meta["contradictions"]:
            md += f"- {c}\n"
        md += "\n"
    md += "---\n"
    for k, v in analysis["clusters"].items():
        md += f"\n### Cluster {int(k)+1} &mdash; {v['tone'].title()}\n"
        md += f"{v['narrative']}\n\n"
        md += f"**Sentiment:** {v['sentiment']} &nbsp;·&nbsp; **Articles:** {v['size']}\n\n"
        if v.get("key_claims"):
            md += "**Key claims:**\n"
            for c in v["key_claims"]:
                md += f"- {c}\n"
    return md

def format_recent_html(recent):
    if not recent:
        return '<div class="recent-empty">Your recent searches will appear here.</div>'
    items = "".join([
        f'<div class="recent-item">{r}</div>'
        for r in reversed(recent[-6:])
    ])
    return f'<div class="recent-list">{items}</div>'

def analyze_topic(topic, days_back, recent):
    if not topic.strip():
        return "", gr.update(visible=False), None, None, [], format_recent_html(recent), recent

    articles = fetch_articles(topic.strip(), days_back=int(days_back))
    if not articles:
        return (
            f"No articles found for **'{topic}'**. Try a broader term or extend the date range.",
            gr.update(visible=False), None, None, [], format_recent_html(recent), recent
        )

    embeddings = embed_and_store(articles)
    articles = cluster_articles(articles, embeddings)
    analysis = analyze_with_llm(articles, topic)
    summary_md = build_summary(analysis, topic)

    cluster_labels = [f"Cluster {int(k)+1}" for k in analysis["clusters"].keys()]
    scores = [v["sentiment_score"] for v in analysis["clusters"].values()]
    fig_s = go.Figure(go.Bar(
        x=cluster_labels, y=scores,
        marker_color=[sentiment_color(s) for s in scores],
        text=[f"{s:+.2f}" for s in scores],
        textposition="outside", width=0.45,
    ))
    fig_s.update_layout(
        title=dict(text="Sentiment per Cluster", font=dict(size=13, color="#111827")),
        yaxis=dict(range=[-1.4, 1.4], gridcolor="#E5E7EB", title="Score", zeroline=False),
        xaxis=dict(gridcolor="#E5E7EB"),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        height=280, margin=dict(l=40, r=20, t=44, b=40),
        font=dict(family="Inter, sans-serif", size=12, color="#374151"),
    )
    fig_s.add_hline(y=0, line_dash="dot", line_color="#D1D5DB", line_width=1)

    df = pd.DataFrame(articles)
    df["date"] = pd.to_datetime(df["published_at"]).dt.date
    daily = df.groupby("date").size().reset_index(name="count")
    fig_t = px.area(daily, x="date", y="count",
                    title="Volume Over Time",
                    color_discrete_sequence=["#2563EB"])
    fig_t.update_traces(line=dict(width=2, color="#2563EB"), fillcolor="rgba(37,99,235,0.08)")
    fig_t.update_layout(
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        height=280, margin=dict(l=40, r=20, t=44, b=40),
        yaxis=dict(title="Articles", gridcolor="#E5E7EB", zeroline=False),
        xaxis=dict(title="", gridcolor="#E5E7EB"),
        font=dict(family="Inter, sans-serif", size=12, color="#374151"),
        title=dict(font=dict(size=13, color="#111827")),
    )

    table_rows = [
        [a["title"][:85], a["source"], a["published_at"][:10], f"Cluster {a['cluster']+1}"]
        for a in articles
    ]

    updated_recent = recent + [topic.strip()]
    return (
        summary_md,
        gr.update(visible=True),
        fig_s, fig_t, table_rows,
        format_recent_html(updated_recent),
        updated_recent,
    )

CSS = """
.gradio-container { max-width: 1280px !important; padding: 0 !important; }
#main-row { gap: 0 !important; min-height: 100vh; }
#sidebar {
    background: #F8FAFC !important;
    border-right: 1px solid #E2E8F0 !important;
    padding: 24px 16px !important;
    min-height: 100vh;
}
.sidebar-logo {
    font-size: 17px; font-weight: 700; color: #111827;
    letter-spacing: -0.3px; margin-bottom: 24px;
    padding-bottom: 16px; border-bottom: 1px solid #E2E8F0;
}
.sidebar-section {
    font-size: 10px; font-weight: 700; letter-spacing: 1.2px;
    text-transform: uppercase; color: #9CA3AF; margin: 20px 0 10px 0;
}
.recent-list { display: flex; flex-direction: column; gap: 4px; }
.recent-item {
    font-size: 12.5px; color: #6B7280; padding: 6px 10px;
    border-radius: 6px; background: transparent;
    border: 1px solid transparent;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.recent-empty { font-size: 12px; color: #D1D5DB; font-style: italic; padding: 4px 2px; }
#sidebar button {
    background: #FFFFFF !important; border: 1px solid #E2E8F0 !important;
    color: #374151 !important; text-align: left !important;
    border-radius: 6px !important; font-size: 12.5px !important;
    padding: 7px 11px !important; margin-bottom: 5px !important;
    width: 100% !important; box-shadow: none !important;
    justify-content: flex-start !important;
}
#sidebar button:hover {
    background: #EEF2FF !important; border-color: #C7D2FE !important;
    color: #2563EB !important;
}
#main-content { padding: 28px 36px !important; }
#input-card {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 12px; padding: 20px 24px; margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
#analyse-btn button {
    background: #2563EB !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    font-size: 14px !important; height: 44px !important;
    width: 100% !important;
}
#analyse-btn button:hover { background: #1D4ED8 !important; }
#results-col {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 12px; padding: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06); margin-top: 4px;
}
#chat-col { padding: 28px 36px !important; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="TrendPulse", css=CSS,
    theme=gr.themes.Soft(
        primary_hue="blue", neutral_hue="slate",
        font=["Inter", "ui-sans-serif", "sans-serif"],
    ),
) as demo:

    recent_state = gr.State([])

    with gr.Tabs():
        with gr.Tab("Analyse Topic"):
            with gr.Row(elem_id="main-row", equal_height=False):

                with gr.Column(scale=1, min_width=210, elem_id="sidebar"):
                    gr.HTML('<div class="sidebar-logo">TrendPulse</div>')
                    gr.HTML('<div class="sidebar-section">Try these</div>')
                    example_btns = [gr.Button(ex, size="sm") for ex in EXAMPLES]
                    gr.HTML('<div class="sidebar-section">Recent</div>')
                    recent_html = gr.HTML(format_recent_html([]))

                with gr.Column(scale=4, elem_id="main-content"):
                    with gr.Group(elem_id="input-card"):
                        with gr.Row(equal_height=True):
                            topic_input = gr.Textbox(
                                label="Topic",
                                placeholder="e.g. OpenAI, Indian economy, climate policy ...",
                                scale=5,
                            )
                            days_slider = gr.Slider(
                                minimum=1, maximum=30, value=7,
                                step=1, label="Days back", scale=1,
                            )
                        with gr.Row(elem_id="analyse-btn"):
                            analyse_btn = gr.Button("Analyse", variant="primary", size="lg")

                    summary_output = gr.Markdown()

                    with gr.Column(visible=False, elem_id="results-col") as results_col:
                        with gr.Row():
                            sentiment_plot = gr.Plot(show_label=False)
                            timeline_plot  = gr.Plot(show_label=False)
                        articles_table = gr.Dataframe(
                            headers=["Title", "Source", "Date", "Cluster"],
                            label="All articles", wrap=True, row_count=8,
                        )

            for btn in example_btns:
                btn.click(fn=lambda x=btn.value: x, outputs=topic_input)

            analyse_btn.click(
                fn=analyze_topic,
                inputs=[topic_input, days_slider, recent_state],
                outputs=[summary_output, results_col, sentiment_plot,
                         timeline_plot, articles_table, recent_html, recent_state],
            )

        with gr.Tab("Chat with Agent"):
            with gr.Row(elem_id="main-row", equal_height=False):
                with gr.Column(scale=1, min_width=210, elem_id="sidebar"):
                    gr.HTML('<div class="sidebar-logo">TrendPulse</div>')
                    gr.HTML('<div class="sidebar-section">What you can ask</div>')
                    gr.HTML("""
                    <div style="font-size:12.5px;color:#6B7280;line-height:2.2;padding:0 2px;">
                        Analyse a topic<br>Compare two topics<br>Find contradictions<br>
                        Track sentiment shifts<br>Summarise coverage<br>Search stored articles
                    </div>""")
                    gr.HTML('<div class="sidebar-section">Example prompts</div>')
                    gr.HTML("""
                    <div style="font-size:12px;color:#9CA3AF;line-height:2.4;padding:0 2px;font-style:italic;">
                        "What narratives exist around AI regulation?"<br>
                        "Compare OpenAI vs DeepMind coverage"<br>
                        "Contradictions in US economy news?"<br>
                        "Summarise latest on Fed rate cuts"
                    </div>""")
                with gr.Column(scale=4, elem_id="chat-col"):
                    gr.ChatInterface(fn=agent_chat, title="")

    gr.HTML("""
    <div style="text-align:center;padding:16px 0 8px;color:#CBD5E1;
                font-size:11px;border-top:1px solid #F1F5F9;margin-top:12px;">
        LangChain &nbsp;·&nbsp; HuggingFace Transformers &nbsp;·&nbsp;
        ChromaDB &nbsp;·&nbsp; Groq LLaMA &nbsp;·&nbsp; Gradio
    </div>""")

if __name__ == "__main__":
    demo.launch(show_api=False)
