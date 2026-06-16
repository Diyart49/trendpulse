import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from src.fetcher import fetch_articles
from src.embedder import embed_and_store
from src.analyzer import cluster_articles, analyze_with_llm
from src.agent import chat as agent_chat

EXAMPLES = ["OpenAI","Indian economy","Climate policy","Fed rate cuts","India-Pakistan",
            "Nvidia","US elections","Bitcoin","Gaza conflict","AI regulation"]

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
            f'background:#F1F5F9;border:1px solid #E2E8F0;margin-bottom:4px;'
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
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
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
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
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


THEME_JS = """
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

CSS = """
/* Variables */
body {
    --bg:#F8FAFC; --surf:#FFFFFF; --surf2:#F1F5F9; --bdr:#E2E8F0;
    --txt:#0F172A; --mid:#374151; --muted:#64748B; --faint:#94A3B8;
    --acc:#2563EB; --acc-bg:#EFF6FF; --acc-bdr:#BFDBFE;
    --sh:rgba(0,0,0,0.06);
}
body.tp-dark {
    --bg:#0D1117; --surf:#161B22; --surf2:#21262D; --bdr:#30363D;
    --txt:#E6EDF3; --mid:#C9D1D9; --muted:#8B949E; --faint:#484F58;
    --acc:#58A6FF; --acc-bg:#0D2137; --acc-bdr:#1F6FEB;
    --sh:rgba(0,0,0,0.3);
}

/* Force theme */
html, body, .gradio-container,
.gradio-container > .main,
.gradio-container > .main > .wrap,
.tabitem, .tab-content, .tabs, div[data-testid] {
    background-color: var(--bg) !important;
    color: var(--txt) !important;
}

/* Strip Gradio chrome */
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

/* Inputs */
input[type=text], input[type=number], textarea {
    background: var(--surf) !important;
    color: var(--txt) !important;
    border: 1.5px solid var(--bdr) !important;
    border-radius: 8px !important;
    font-size: 14px !important;
}
input[type=text]:focus, textarea:focus {
    border-color: var(--acc) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}

/* Fix label highlight bug */
label, label span, .label-wrap, .label-wrap span {
    color: var(--muted) !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    padding: 0 0 4px 0 !important;
    user-select: none !important;
}

/* Tab nav */
.tab-nav {
    background: var(--surf) !important;
    border-bottom: 1px solid var(--bdr) !important;
    padding: 0 16px !important;
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
    overflow-x: auto !important;
}
.tab-nav button {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: var(--faint) !important;
    padding: 12px 18px !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    white-space: nowrap !important;
    flex-shrink: 0 !important;
    min-width: fit-content !important;
}
.tab-nav button.selected {
    color: var(--acc) !important;
    border-bottom: 2px solid var(--acc) !important;
}

/* Sticky sidebar */
#tp-sidebar {
    position: sticky !important;
    top: 0 !important;
    height: 100vh !important;
    overflow-y: auto !important;
    background: var(--surf) !important;
    border-right: 1px solid var(--bdr) !important;
    padding: 20px 14px !important;
    flex-shrink: 0 !important;
}
#tp-sidebar button {
    background: var(--surf2) !important;
    border: 1px solid var(--bdr) !important;
    color: var(--mid) !important;
    text-align: left !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    padding: 7px 11px !important;
    margin-bottom: 5px !important;
    width: 100% !important;
    box-shadow: none !important;
    justify-content: flex-start !important;
    font-weight: 400 !important;
    transition: all 0.15s !important;
}
#tp-sidebar button:hover {
    background: var(--acc-bg) !important;
    color: var(--acc) !important;
    border-color: var(--acc-bdr) !important;
}

/* Main & chat areas */
#tp-main {
    background: var(--bg) !important;
    padding: 24px 32px !important;
    min-height: 100vh !important;
}
#tp-chat {
    background: var(--bg) !important;
    padding: 24px 32px !important;
}

/* Hero */
#tp-hero {
    background: linear-gradient(135deg,#EFF6FF 0%,#F8FAFC 55%,#F0FDF4 100%) !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 14px !important;
    padding: 24px 28px !important;
    margin-bottom: 18px !important;
}
body.tp-dark #tp-hero {
    background: linear-gradient(135deg,#0D2137 0%,#0D1117 55%,#0D1A0D 100%) !important;
    border-color: #30363D !important;
}

/* Search card */
#tp-search {
    background: var(--surf) !important;
    border: 1.5px solid var(--bdr) !important;
    border-radius: 14px !important;
    padding: 20px 22px !important;
    box-shadow: 0 2px 8px var(--sh) !important;
    margin-bottom: 18px !important;
}
#tp-search > div, #tp-search .gr-group, #tp-search .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Analyse button */
#tp-btn button {
    background: linear-gradient(135deg,#2563EB,#1D4ED8) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    color: #FFFFFF !important;
    height: 44px !important;
    width: 100% !important;
    margin-top: 12px !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.3) !important;
    transition: all 0.2s !important;
}
#tp-btn button:hover {
    background: linear-gradient(135deg,#1D4ED8,#1E40AF) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.45) !important;
}

/* Results */
#tp-results {
    background: var(--surf) !important;
    border: 1px solid var(--bdr) !important;
    border-radius: 14px !important;
    padding: 24px !important;
    box-shadow: 0 2px 8px var(--sh) !important;
    margin-top: 8px !important;
}

/* Markdown */
.prose h2, .md h2 { color:var(--txt) !important; font-size:20px !important; font-weight:700 !important; }
.prose h3, .md h3 { color:var(--txt) !important; font-size:15px !important; font-weight:600 !important; }
.prose p, .md p, .prose li, .md li { color:var(--mid) !important; line-height:1.7 !important; }
.prose strong, .md strong { color:var(--txt) !important; }
.prose blockquote, .md blockquote {
    border-left: 3px solid var(--acc) !important;
    background: var(--acc-bg) !important;
    padding: 10px 14px !important;
    border-radius: 0 8px 8px 0 !important;
    color: var(--acc) !important;
    font-style: italic !important;
    margin: 12px 0 !important;
}
.prose hr, .md hr { border-color: var(--bdr) !important; }

/* Table */
.gr-dataframe th, thead th {
    background: var(--surf2) !important; color: var(--muted) !important;
    font-size: 11px !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.5px !important;
    padding: 9px 12px !important;
}
.gr-dataframe td, tbody td {
    font-size: 13px !important; color: var(--mid) !important;
    background: var(--surf) !important;
    border-top: 1px solid var(--bdr) !important;
    padding: 9px 12px !important;
}

/* Slider */
input[type=range] { accent-color: var(--acc) !important; }

/* Dark mode button — fixed top right */
#tp-theme-fixed {
    position: fixed !important;
    top: 14px !important;
    right: 16px !important;
    z-index: 9999 !important;
}
#tp-theme-fixed button {
    background: var(--surf) !important;
    border: 1px solid var(--bdr) !important;
    color: var(--mid) !important;
    border-radius: 20px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 5px 16px !important;
    height: 30px !important;
    box-shadow: 0 1px 4px var(--sh) !important;
    width: auto !important;
    min-width: 90px !important;
    transition: all 0.15s !important;
}
#tp-theme-fixed button:hover {
    background: var(--acc-bg) !important;
    color: var(--acc) !important;
    border-color: var(--acc-bdr) !important;
}

footer { display: none !important; }
"""

HERO_HTML = """
<div id="tp-hero">
    <div style="font-size:22px;font-weight:800;color:#0F172A;letter-spacing:-0.5px;margin-bottom:6px;">
        News Intelligence
    </div>
    <div style="font-size:13.5px;color:#64748B;line-height:1.7;max-width:500px;">
        Analyse live news for any topic &mdash; track narratives, detect sentiment shifts,
        and surface contradictions across sources.
    </div>
    <div style="display:flex;gap:20px;margin-top:14px;flex-wrap:wrap;">
        <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#475569;">
            <div style="width:7px;height:7px;border-radius:50%;background:#16A34A;flex-shrink:0;"></div>
            Narrative clustering
        </div>
        <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#475569;">
            <div style="width:7px;height:7px;border-radius:50%;background:#3B82F6;flex-shrink:0;"></div>
            Sentiment analysis
        </div>
        <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#475569;">
            <div style="width:7px;height:7px;border-radius:50%;background:#F59E0B;flex-shrink:0;"></div>
            Contradiction detection
        </div>
    </div>
</div>
"""

SIDEBAR_HEADER = """
<div style="font-size:15px;font-weight:700;color:#0F172A;padding-bottom:14px;
    border-bottom:1px solid #E2E8F0;margin-bottom:16px;letter-spacing:-0.2px;">
    TrendPulse
</div>
<div style="font-size:10px;font-weight:700;letter-spacing:1.2px;
    text-transform:uppercase;color:#94A3B8;margin-bottom:10px;">
    Try these topics
</div>
"""

caps = ["Analyse any topic live","Compare two topics","Find contradictions",
        "Summarise coverage","Search stored articles"]
CAP_ITEMS = "".join([
    f'<div style="font-size:13px;color:#475569;padding:7px 11px;'
    f'background:#F8FAFC;border:1px solid #E2E8F0;'
    f'border-radius:8px;margin-bottom:5px;">{c}</div>'
    for c in caps
])

CHAT_SIDEBAR = f"""
<div style="font-size:15px;font-weight:700;color:#0F172A;padding-bottom:14px;
    border-bottom:1px solid #E2E8F0;margin-bottom:16px;">TrendPulse</div>
<div style="font-size:10px;font-weight:700;letter-spacing:1.2px;
    text-transform:uppercase;color:#94A3B8;margin-bottom:12px;">
    What the agent can do
</div>
{CAP_ITEMS}
<div style="font-size:10px;font-weight:700;letter-spacing:1.2px;
    text-transform:uppercase;color:#94A3B8;margin:20px 0 10px;">
    Example prompts
</div>
<div style="font-size:12px;color:#94A3B8;line-height:2.4;font-style:italic;">
    "Narratives around AI regulation?"<br>
    "Compare OpenAI vs DeepMind"<br>
    "Contradictions in US economy?"<br>
    "Summarise Fed rate cut news"
</div>
"""

with gr.Blocks(
    title="TrendPulse", css=CSS,
    theme=gr.themes.Base(
        primary_hue="blue", neutral_hue="slate",
        font=["Inter","ui-sans-serif","sans-serif"]
    )
) as demo:

    recent_state = gr.State([])

    # Fixed dark mode toggle — top right
    with gr.Row(elem_id="tp-theme-fixed"):
        dark_btn = gr.Button("Dark mode", size="sm")

    with gr.Tabs():

        with gr.Tab("Analyse Topic"):
            with gr.Row(equal_height=False):

                with gr.Column(scale=1, min_width=200, elem_id="tp-sidebar"):
                    gr.HTML(SIDEBAR_HEADER)
                    example_btns = [gr.Button(ex, size="sm") for ex in EXAMPLES]
                    gr.HTML("""
                    <div style="font-size:10px;font-weight:700;letter-spacing:1.2px;
                        text-transform:uppercase;color:#94A3B8;margin:20px 0 10px;">
                        Recent searches
                    </div>
                    """)
                    recent_html = gr.HTML(
                        '<p style="font-size:12px;color:#94A3B8;'
                        'font-style:italic;margin:0;">No recent searches yet.</p>'
                    )

                with gr.Column(scale=4, elem_id="tp-main"):
                    gr.HTML(HERO_HTML)
                    with gr.Group(elem_id="tp-search"):
                        topic_input = gr.Textbox(
                            label="Topic",
                            placeholder="e.g. OpenAI, Indian economy, climate policy ...",
                        )
                        days_slider = gr.Slider(
                            minimum=1, maximum=30, value=7,
                            step=1, label="Days back",
                        )
                        with gr.Row(elem_id="tp-btn"):
                            analyse_btn = gr.Button("Analyse", variant="primary", size="lg")

                    summary_md = gr.Markdown()

                    with gr.Column(visible=False, elem_id="tp-results") as results_col:
                        with gr.Row():
                            sentiment_plot = gr.Plot(show_label=False)
                            timeline_plot = gr.Plot(show_label=False)
                        gr.HTML('<div style="height:10px"></div>')
                        articles_table = gr.Dataframe(
                            headers=["Title","Source","Date","Cluster"],
                            label="All articles", wrap=True, row_count=8,
                        )

            for btn in example_btns:
                btn.click(fn=lambda x=btn.value: x, outputs=topic_input)

            dark_btn.click(fn=None, js=THEME_JS)

            analyse_btn.click(
                fn=analyze_topic,
                inputs=[topic_input, days_slider, recent_state],
                outputs=[summary_md, results_col, sentiment_plot,
                         timeline_plot, articles_table, recent_html, recent_state],
            )

        with gr.Tab("Agent Chat"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=200, elem_id="tp-sidebar"):
                    gr.HTML(CHAT_SIDEBAR)
                with gr.Column(scale=4, elem_id="tp-chat"):
                    gr.ChatInterface(fn=agent_chat, title="")

if __name__ == "__main__":
    demo.launch(show_api=False)
