import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from src.fetcher import fetch_articles
from src.embedder import embed_and_store
from src.analyzer import cluster_articles, analyze_with_llm
from src.agent import chat as agent_chat

EXAMPLES = ["OpenAI", "Indian economy", "Climate policy", "Fed rate cuts", "India-Pakistan",
            "Nvidia", "US elections", "Bitcoin", "Gaza conflict", "AI regulation"]

def sentiment_color(score):
    if score > 0.1: return "#16A34A"
    if score < -0.1: return "#DC2626"
    return "#6B7280"

def build_summary(analysis, topic):
    meta = analysis["meta"]
    md = f"## {topic}\n\n"
    md += f"**{analysis['total_articles']} articles analysed** &nbsp;&middot;&nbsp; "
    md += f"**{meta['overall_sentiment'].title()} sentiment**\n\n"
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
        md += f"**Sentiment:** {v['sentiment']} &nbsp;&middot;&nbsp; **Articles:** {v['size']}\n\n"
        if v.get("key_claims"):
            md += "**Key claims:**\n"
            for c in v["key_claims"]:
                md += f"- {c}\n"
    return md

def format_recent(recent):
    if not recent:
        return '<p style="font-size:12px;color:#94A3B8;font-style:italic;margin:0;">No recent searches yet.</p>'
    items = ""
    for r in reversed(recent[-6:]):
        items += (
            f'<div style="font-size:13px;color:#64748B;padding:6px 10px;border-radius:6px;'
            f'background:var(--surf2);border:1px solid var(--bdr);margin-bottom:4px;'
            f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{r}</div>'
        )
    return items

def analyze_topic(topic, days_back, recent):
    if not topic.strip():
        return "", gr.update(visible=False), None, None, [], format_recent(recent), recent
    articles = fetch_articles(topic.strip(), days_back=int(days_back))
    if not articles:
        return (
            f"No articles found for **'{topic}'**. Try a broader term.",
            gr.update(visible=False), None, None, [], format_recent(recent), recent
        )
    embeddings = embed_and_store(articles)
    articles = cluster_articles(articles, embeddings)
    analysis = analyze_with_llm(articles, topic)

    cluster_labels = [f"Cluster {int(k)+1}" for k in analysis["clusters"].keys()]
    scores = [v["sentiment_score"] for v in analysis["clusters"].values()]

    fig_s = go.Figure(go.Bar(
        x=cluster_labels, y=scores,
        marker_color=[sentiment_color(s) for s in scores],
        text=[f"{s:+.2f}" for s in scores],
        textposition="outside", width=0.45,
    ))
    fig_s.update_layout(
        title=dict(text="Sentiment per Cluster", font=dict(size=13, color="#1E293B")),
        yaxis=dict(range=[-1.4,1.4], gridcolor="#F1F5F9", title="Score", zeroline=False, color="#64748B"),
        xaxis=dict(gridcolor="#F1F5F9", color="#64748B"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=280, margin=dict(l=40,r=20,t=44,b=40),
        font=dict(family="Inter,sans-serif", size=12, color="#374151"),
    )
    fig_s.add_hline(y=0, line_dash="dot", line_color="#E2E8F0", line_width=1.5)

    df = pd.DataFrame(articles)
    df["date"] = pd.to_datetime(df["published_at"]).dt.date
    daily = df.groupby("date").size().reset_index(name="count")
    fig_t = px.area(daily, x="date", y="count", title="Article Volume",
                    color_discrete_sequence=["#3B82F6"])
    fig_t.update_traces(line=dict(width=2, color="#3B82F6"), fillcolor="rgba(59,130,246,0.07)")
    fig_t.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=280, margin=dict(l=40,r=20,t=44,b=40),
        yaxis=dict(title="Articles", gridcolor="#F1F5F9", zeroline=False, color="#64748B"),
        xaxis=dict(title="", gridcolor="#F1F5F9", color="#64748B"),
        font=dict(family="Inter,sans-serif", size=12, color="#374151"),
        title=dict(font=dict(size=13, color="#1E293B")),
    )

    table_rows = [
        [a["title"][:85], a["source"], a["published_at"][:10], f"Cluster {a['cluster']+1}"]
        for a in articles
    ]
    updated = recent + [topic.strip()]
    return (
        build_summary(analysis, topic),
        gr.update(visible=True),
        fig_s, fig_t, table_rows,
        format_recent(updated), updated,
    )

# Robust JS snippet targeting the core body object directly
THEME_JS = """
() => {
    const body = document.body;
    const isDark = body.classList.toggle('tp-dark');
    const buttons = document.querySelectorAll('.theme-toggle-btn button');
    buttons.forEach(btn => {
        btn.textContent = isDark ? '✨ Light mode' : '🌙 Dark mode';
    });
}
"""

CSS = """
body {
    --bg:#F8FAFC; --surf:#FFFFFF; --surf2:#F1F5F9; --bdr:#E2E8F0;
    --txt:#0F172A; --mid:#374151; --muted:#64748B; --faint:#94A3B8;
    --acc:#2563EB; --acc-bg:#EFF6FF; --acc-bdr:#BFDBFE;
    --sh:rgba(0,0,0,0.04);
}
body.tp-dark {
    --bg:#0D1117; --surf:#161B22; --surf2:#21262D; --bdr:#30363D;
    --txt:#E6EDF3; --mid:#C9D1D9; --muted:#8B949E; --faint:#484F58;
    --acc:#58A6FF; --acc-bg:#0D2137; --acc-bdr:#1F6FEB;
    --sh:rgba(0,0,0,0.3);
}

html, body, .gradio-container, .gradio-container > .main, .tabitem, .tab-content, .tabs {
    background-color: var(--bg) !important;
    color: var(--txt) !important;
}

/* Remove default Gradio Card visual wrappers */
.block, .form, .gap, .padded, .gr-group, .gr-box {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

.gradio-container {
    font-family: 'Inter', -apple-system, sans-serif !important;
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

input[type=text], textarea {
    background: var(--surf) !important;
    color: var(--txt) !important;
    border: 1.5px solid var(--bdr) !important;
    border-radius: 8px !important;
    font-size: 14px !important;
}

/* Clean tab selection links */
.tab-nav {
    background: var(--surf) !important;
    border-bottom: 1px solid var(--bdr) !important;
    padding: 0 24px !important;
}
.tab-nav button {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 14px 20px !important;
}
.tab-nav button.selected {
    color: var(--acc) !important;
    border-bottom: 2px solid var(--acc) !important;
}

#tp-sidebar {
    position: sticky !important;
    top: 0 !important;
    height: 100vh !important;
    overflow-y: auto !important;
    background: var(--surf) !important;
    border-right: 1px solid var(--bdr) !important;
    padding: 24px 16px !important;
}
#tp-sidebar button {
    background: var(--surf2) !important;
    border: 1px solid var(--bdr) !important;
    color: var(--mid) !important;
    text-align: left !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    padding: 8px 12px !important;
    margin-bottom: 6px !important;
    width: 100% !important;
    justify-content: flex-start !important;
    transition: all 0.1s ease;
}
#tp-sidebar button:hover {
    background: var(--acc-bg) !important;
    color: var(--acc) !important;
    border-color: var(--acc-bdr) !important;
}

#tp-main, #tp-chat {
    background: var(--bg) !important;
    padding: 32px 40px !important;
    min-height: 100vh !important;
}

#tp-hero {
    background: linear-gradient(135deg, var(--acc-bg) 0%, var(--surf) 100%) !important;
    border: 1px solid var(--bdr) !important;
    border-radius: 12px !important;
    padding: 24px !important;
    margin-bottom: 20px !important;
}

#tp-search {
    background: var(--surf) !important;
    border: 1px solid var(--bdr) !important;
    border-radius: 12px !important;
    padding: 24px !important;
    box-shadow: 0 4px 12px var(--sh) !important;
    margin-bottom: 20px !important;
}

#tp-btn button {
    background: var(--acc) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    color: #FFFFFF !important;
    height: 42px !important;
    width: 100% !important;
    margin-top: 12px !important;
}

#tp-results {
    background: var(--surf) !important;
    border: 1px solid var(--bdr) !important;
    border-radius: 12px !important;
    padding: 24px !important;
    margin-top: 12px !important;
}

.theme-toggle-fixed {
    position: fixed !important;
    top: 10px !important;
    right: 20px !important;
    z-index: 99999 !important;
}
.theme-toggle-fixed button {
    background: var(--surf) !important;
    border: 1px solid var(--bdr) !important;
    color: var(--mid) !important;
    border-radius: 20px !important;
    font-size: 12px !important;
    padding: 4px 14px !important;
    width: auto !important;
}

footer { display: none !important; }
"""

HERO_HTML = """
<div id="tp-hero">
    <div style="font-size:22px;font-weight:800;color:var(--txt);letter-spacing:-0.5px;margin-bottom:6px;">
        News Intelligence Dashboard
    </div>
    <div style="font-size:13.5px;color:var(--muted);line-height:1.7;max-width:600px;">
        Analyse modern narrative clusters, map underlying macro trends, and track global sentiment distribution across multiple dimensions.
    </div>
</div>
"""

SIDEBAR_HEADER = """
<div style="font-size:16px;font-weight:800;color:var(--txt);padding-bottom:12px;
    border-bottom:1px solid var(--bdr);margin-bottom:16px;letter-spacing:-0.3px;">
    TrendPulse
</div>
<div style="font-size:10px;font-weight:700;letter-spacing:1px;
    text-transform:uppercase;color:var(--faint);margin-bottom:12px;">
    Suggested Topics
</div>
"""

CHAT_SIDEBAR = """
<div style="font-size:16px;font-weight:800;color:var(--txt);padding-bottom:12px;
    border-bottom:1px solid var(--bdr);margin-bottom:16px;letter-spacing:-0.3px;">
    TrendPulse
</div>
<div style="font-size:10px;font-weight:700;letter-spacing:1px;
    text-transform:uppercase;color:var(--faint);margin-bottom:12px;">
    Analytical Agent Desk
</div>
<p style="font-size:13px; color:var(--muted); line-height:1.6;">
    Query contextual extractions, run custom cross-examinations, or compare divergent narratives.
</p>
"""

with gr.Blocks(title="TrendPulse", css=CSS, theme=gr.themes.Base()) as demo:
    recent_state = gr.State([])

    # Theme toggler globally pinned at top-right
    with gr.Row(elem_classes="theme-toggle-fixed"):
        dark_btn = gr.Button("🌙 Dark mode", size="sm", elem_classes="theme-toggle-btn")

    with gr.Tabs():
        with gr.Tab("Analyse Topic"):
            with gr.Row(equal_height=False):
                # SIDEBAR
                with gr.Column(scale=1, min_width=240, elem_id="tp-sidebar"):
                    gr.HTML(SIDEBAR_HEADER)
                    
                    # Create example shortcuts mapping
                    example_btns = []
                    for ex in EXAMPLES:
                        example_btns.append(gr.Button(ex, size="sm"))
                        
                    gr.HTML('<div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--faint);margin:24px 0 10px;">Recent Searches</div>')
                    recent_html = gr.HTML('<p style="font-size:12px;color:var(--faint);font-style:italic;margin:0;">No recent searches yet.</p>')

                # MAIN LAB AREA
                with gr.Column(scale=4, elem_id="tp-main"):
                    gr.HTML(HERO_HTML)
                    with gr.Group(elem_id="tp-search"):
                        topic_input = gr.Textbox(
                            label="Target Topic Definition",
                            placeholder="Type industry keyword or market event..."
                        )
                        days_slider = gr.Slider(
                            minimum=1, maximum=30, value=7,
                            step=1, label="Historical Windows (Days Back)"
                        )
                        with gr.Row(elem_id="tp-btn"):
                            analyse_btn = gr.Button("Execute Analysis", variant="primary", size="lg")

                    summary_md = gr.Markdown()

                    with gr.Column(visible=False, elem_id="tp-results") as results_col:
                        with gr.Row():
                            sentiment_plot = gr.Plot(show_label=False)
                            timeline_plot = gr.Plot(show_label=False)
                        gr.HTML('<div style="height:14px"></div>')
                        articles_table = gr.Dataframe(
                            headers=["Title", "Source", "Date", "Cluster"],
                            label="Consolidated Stream Records", wrap=True, row_count=8
                        )

        with gr.Tab("Agent Chat"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=240, elem_id="tp-sidebar"):
                    gr.HTML(CHAT_SIDEBAR)
                with gr.Column(scale=4, elem_id="tp-chat"):
                    gr.ChatInterface(fn=agent_chat, title="")

    # WIRE EVENTS CLEANLY
    for btn in example_btns:
        btn.click(fn=lambda x: x, inputs=[btn], outputs=[topic_input])

    dark_btn.click(fn=None, js=THEME_JS)

    analyse_btn.click(
        fn=analyze_topic,
        inputs=[topic_input, days_slider, recent_state],
        outputs=[summary_md, results_col, sentiment_plot, timeline_plot, articles_table, recent_html, recent_state],
    )

if __name__ == "__main__":
    demo.launch(show_api=False)
