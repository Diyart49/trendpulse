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
    md = f"## {topic}\n\n**{analysis['total_articles']} articles analysed** &nbsp;·&nbsp; **{meta['overall_sentiment'].title()} sentiment**\n\n> {meta['dominant_narrative']}\n\n"
    if meta.get("contradictions"):
        md += "**Contradictions detected:**\n" + "".join([f"- {c}\n" for c in meta["contradictions"]]) + "\n"
    md += "---\n"
    for k, v in analysis["clusters"].items():
        md += f"\n### Cluster {int(k)+1} &mdash; {v['tone'].title()}\n{v['narrative']}\n\n**Sentiment:** {v['sentiment']} &nbsp;·&nbsp; **Articles:** {v['size']}\n\n"
        if v.get("key_claims"):
            md += "**Key claims:**\n" + "".join([f"- {c}\n" for c in v["key_claims"]])
    return md

def format_recent(recent):
    if not recent:
        return '<p style="font-size:12px;color:#94A3B8;font-style:italic;margin:0;">No recent searches yet.</p>'
    return "".join([
        f'<div style="font-size:13px;color:#64748B;padding:6px 10px;border-radius:6px;background:#F1F5F9;border:1px solid #E2E8F0;margin-bottom:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{r}</div>'
        for r in reversed(recent[-6:])
    ])

def analyze_topic(topic, days_back, recent):
    if not topic.strip():
        return "", gr.update(visible=False), None, None, [], format_recent(recent), recent
    articles = fetch_articles(topic.strip(), days_back=int(days_back))
    if not articles:
        return f"No articles found for **'{topic}'**. Try a broader term.", gr.update(visible=False), None, None, [], format_recent(recent), recent
    embeddings = embed_and_store(articles)
    articles = cluster_articles(articles, embeddings)
    analysis = analyze_with_llm(articles, topic)
    cluster_labels = [f"Cluster {int(k)+1}" for k in analysis["clusters"].keys()]
    scores = [v["sentiment_score"] for v in analysis["clusters"].values()]
    fig_s = go.Figure(go.Bar(x=cluster_labels, y=scores, marker_color=[sentiment_color(s) for s in scores], text=[f"{s:+.2f}" for s in scores], textposition="outside", width=0.45))
    fig_s.update_layout(title=dict(text="Sentiment per Cluster", font=dict(size=13, color="#1E293B")), yaxis=dict(range=[-1.4,1.4], gridcolor="#F1F5F9", title="Score", zeroline=False, color="#64748B"), xaxis=dict(gridcolor="#F1F5F9", color="#64748B"), plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", height=280, margin=dict(l=40,r=20,t=44,b=40), font=dict(family="Inter, sans-serif", size=12, color="#374151"))
    fig_s.add_hline(y=0, line_dash="dot", line_color="#E2E8F0", line_width=1.5)
    df = pd.DataFrame(articles)
    df["date"] = pd.to_datetime(df["published_at"]).dt.date
    daily = df.groupby("date").size().reset_index(name="count")
    fig_t = px.area(daily, x="date", y="count", title="Article Volume", color_discrete_sequence=["#3B82F6"])
    fig_t.update_traces(line=dict(width=2, color="#3B82F6"), fillcolor="rgba(59,130,246,0.07)")
    fig_t.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", height=280, margin=dict(l=40,r=20,t=44,b=40), yaxis=dict(title="Articles", gridcolor="#F1F5F9", zeroline=False, color="#64748B"), xaxis=dict(title="", gridcolor="#F1F5F9", color="#64748B"), font=dict(family="Inter, sans-serif", size=12, color="#374151"), title=dict(font=dict(size=13, color="#1E293B")))
    table_rows = [[a["title"][:85], a["source"], a["published_at"][:10], f"Cluster {a['cluster']+1}"] for a in articles]
    updated = recent + [topic.strip()]
    return build_summary(analysis, topic), gr.update(visible=True), fig_s, fig_t, table_rows, format_recent(updated), updated

CSS = """
:root { --bg:#F8FAFC; --surface:#FFFFFF; --surface2:#F1F5F9; --border:#E2E8F0; --text:#0F172A; --text-mid:#374151; --text-muted:#64748B; --muted:#94A3B8; --accent:#2563EB; --accent-dim:#EFF6FF; }
html[data-theme="dark"] { --bg:#0F172A; --surface:#1E293B; --surface2:#334155; --border:#334155; --text:#F1F5F9; --text-mid:#CBD5E1; --text-muted:#94A3B8; --muted:#475569; --accent:#3B82F6; --accent-dim:#1E3A5F; }
html, body, .gradio-container, .main, .wrap, [data-testid], .tabitem, .tab-content, .tabs { background: var(--bg) !important; color: var(--text) !important; }
.block, .form, .label-wrap, .container { background: transparent !important; border-color: var(--border) !important; }
input, textarea, select { background: var(--surface) !important; color: var(--text) !important; border-color: var(--border) !important; }
label, .label-wrap span { color: var(--text-muted) !important; font-size: 13px !important; font-weight: 500 !important; }
.gradio-container { max-width: 1300px !important; margin: 0 auto !important; padding: 0 !important; font-family: 'Inter', sans-serif !important; }
.tab-nav { background: var(--surface) !important; border-bottom: 1px solid var(--border) !important; padding: 0 24px !important; margin: 0 !important; }
.tab-nav button { font-size: 13px !important; font-weight: 600 !important; color: var(--muted) !important; padding: 14px 16px !important; background: transparent !important; border: none !important; border-bottom: 2px solid transparent !important; }
.tab-nav button.selected { color: var(--accent) !important; border-bottom-color: var(--accent) !important; }
#sidebar { background: var(--surface) !important; border-right: 1px solid var(--border) !important; padding: 24px 16px !important; min-height: calc(100vh - 52px) !important; }
#sidebar button { background: var(--surface2) !important; border: 1px solid var(--border) !important; color: var(--text-mid) !important; text-align: left !important; border-radius: 8px !important; font-size: 13px !important; padding: 8px 12px !important; margin-bottom: 5px !important; width: 100% !important; box-shadow: none !important; justify-content: flex-start !important; font-weight: 400 !important; }
#sidebar button:hover { background: var(--accent-dim) !important; color: var(--accent) !important; }
#main-area { background: var(--bg) !important; padding: 28px 36px !important; }
#search-card { background: var(--surface) !important; border: 1.5px solid var(--border) !important; border-radius: 14px !important; padding: 20px 24px 18px !important; box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important; margin-bottom: 20px !important; }
#analyse-btn button { background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; font-size: 14px !important; color: #FFFFFF !important; height: 44px !important; width: 100% !important; box-shadow: 0 2px 8px rgba(37,99,235,0.25) !important; margin-top: 10px !important; }
#analyse-btn button:hover { background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%) !important; transform: translateY(-1px) !important; }
#results-card { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 14px !important; padding: 24px !important; box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important; margin-top: 4px !important; }
.prose h2, .md h2 { color: var(--text) !important; font-size: 20px !important; font-weight: 700 !important; }
.prose h3, .md h3 { color: var(--text) !important; font-size: 15px !important; font-weight: 600 !important; }
.prose blockquote, .md blockquote { border-left: 3px solid var(--accent) !important; background: var(--accent-dim) !important; padding: 12px 16px !important; border-radius: 0 8px 8px 0 !important; color: var(--accent) !important; }
.gr-dataframe th, thead th { background: var(--surface2) !important; color: var(--text-muted) !important; font-size: 11px !important; font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 0.6px !important; }
.gr-dataframe td, tbody td { font-size: 13px !important; color: var(--text-mid) !important; background: var(--surface) !important; border-top: 1px solid var(--border) !important; }
input[type=range] { accent-color: var(--accent) !important; }
#chat-area { background: var(--bg) !important; padding: 28px 36px !important; }
footer { display: none !important; }
"""

TOPBAR = """
<div style="background:var(--surface,#fff);border-bottom:1px solid var(--border,#E2E8F0);padding:0 28px;display:flex;align-items:center;justify-content:space-between;height:52px;position:sticky;top:0;z-index:100;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
    <div style="font-size:15px;font-weight:700;color:var(--text,#0F172A);letter-spacing:-0.2px;">TrendPulse</div>
    <div style="font-size:11px;color:var(--muted,#94A3B8);letter-spacing:0.3px;">LangChain &nbsp;·&nbsp; HuggingFace &nbsp;·&nbsp; ChromaDB &nbsp;·&nbsp; Groq</div>
    <button id="tp-theme-btn" onclick="(function(btn){
        const html=document.documentElement;
        const dark=html.getAttribute('data-theme')==='dark';
        if(dark){html.removeAttribute('data-theme');btn.textContent='Dark mode';}
        else{html.setAttribute('data-theme','dark');btn.textContent='Light mode';}
    })(this)" style="background:var(--surface2,#F1F5F9);border:1px solid var(--border,#E2E8F0);color:var(--text-mid,#374151);border-radius:8px;font-size:12.5px;font-weight:500;padding:6px 14px;cursor:pointer;font-family:inherit;transition:all 0.2s;">
        Dark mode
    </button>
</div>
"""

HERO = """
<div style="background:linear-gradient(135deg,#EFF6FF 0%,#F8FAFC 60%,#F0FDF4 100%);border:1px solid #E2E8F0;border-radius:14px;padding:28px 32px;margin-bottom:20px;">
    <div style="font-size:24px;font-weight:800;color:#0F172A;letter-spacing:-0.5px;margin-bottom:8px;">News Intelligence</div>
    <div style="font-size:14px;color:#64748B;line-height:1.7;max-width:520px;">Analyse live news coverage for any topic — track narratives, detect sentiment shifts, and surface contradictions across sources in real time.</div>
    <div style="display:flex;gap:20px;margin-top:16px;flex-wrap:wrap;">
        <div style="display:flex;align-items:center;gap:7px;font-size:12.5px;color:#475569;"><div style="width:8px;height:8px;border-radius:50%;background:#16A34A;flex-shrink:0;"></div>Narrative clustering</div>
        <div style="display:flex;align-items:center;gap:7px;font-size:12.5px;color:#475569;"><div style="width:8px;height:8px;border-radius:50%;background:#3B82F6;flex-shrink:0;"></div>Sentiment analysis</div>
        <div style="display:flex;align-items:center;gap:7px;font-size:12.5px;color:#475569;"><div style="width:8px;height:8px;border-radius:50%;background:#F59E0B;flex-shrink:0;"></div>Contradiction detection</div>
    </div>
</div>
"""

with gr.Blocks(title="TrendPulse", css=CSS, theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate", font=["Inter","ui-sans-serif","sans-serif"])) as demo:
    recent_state = gr.State([])
    gr.HTML(TOPBAR)

    with gr.Tabs():
        with gr.Tab("Analyse Topic"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=200, elem_id="sidebar"):
                    gr.HTML('<div style="font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:#94A3B8;margin:4px 0 12px 0;">Try these topics</div>')
                    example_btns = [gr.Button(ex, size="sm") for ex in EXAMPLES]
                    gr.HTML('<div style="font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:#94A3B8;margin:22px 0 10px 0;">Recent searches</div>')
                    recent_html = gr.HTML('<p style="font-size:12px;color:#94A3B8;font-style:italic;margin:0;">No recent searches yet.</p>')

                with gr.Column(scale=4, elem_id="main-area"):
                    gr.HTML(HERO)
                    with gr.Group(elem_id="search-card"):
                        with gr.Row(equal_height=True):
                            topic_input = gr.Textbox(label="Topic", placeholder="e.g. OpenAI, Indian economy, climate policy ...", scale=4, elem_id="topic-input")
                            days_slider = gr.Slider(minimum=1, maximum=30, value=7, step=1, label="Days back", scale=1, min_width=150)
                        with gr.Row(elem_id="analyse-btn"):
                            analyse_btn = gr.Button("Analyse", variant="primary", size="lg")
                    summary_output = gr.Markdown()
                    with gr.Column(visible=False, elem_id="results-card") as results_col:
                        with gr.Row():
                            sentiment_plot = gr.Plot(show_label=False)
                            timeline_plot  = gr.Plot(show_label=False)
                        gr.HTML('<div style="height:12px;"></div>')
                        articles_table = gr.Dataframe(headers=["Title","Source","Date","Cluster"], label="All articles", wrap=True, row_count=8)

            for btn in example_btns:
                btn.click(fn=lambda x=btn.value: x, outputs=topic_input)
            analyse_btn.click(fn=analyze_topic, inputs=[topic_input, days_slider, recent_state], outputs=[summary_output, results_col, sentiment_plot, timeline_plot, articles_table, recent_html, recent_state])

        with gr.Tab("Chat with Agent"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=200, elem_id="sidebar"):
                    caps = ["Analyse any topic live","Compare two topics","Find contradictions","Summarise coverage","Search stored articles"]
                    items = "".join([f'<div style="font-size:13px;color:#475569;padding:8px 12px;background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;margin-bottom:6px;">{c}</div>' for c in caps])
                    gr.HTML(f'<div style="font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:#94A3B8;margin:4px 0 12px 0;">What the agent can do</div>{items}<div style="font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:#94A3B8;margin:20px 0 10px 0;">Example prompts</div><div style="font-size:12.5px;color:#94A3B8;line-height:2.3;font-style:italic;">"Narratives around AI regulation?"<br>"Compare OpenAI vs DeepMind"<br>"Contradictions in US economy?"<br>"Summarise Fed rate cut news"</div>')
                with gr.Column(scale=4, elem_id="chat-area"):
                    gr.ChatInterface(fn=agent_chat, title="")

if __name__ == "__main__":
    demo.launch(show_api=False)
