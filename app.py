
import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from src.fetcher import fetch_articles
from src.embedder import embed_and_store
from src.analyzer import cluster_articles, analyze_with_llm
from src.agent import chat as agent_chat

EXAMPLES = ["OpenAI","Indian economy","Climate policy","Fed rate cuts","India-Pakistan","Nvidia","US elections","Bitcoin","Gaza conflict","AI regulation"]

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
        md += f"\n### Cluster {int(k)+1} — {v['tone'].title()}\n{v['narrative']}\n\n**Sentiment:** {v['sentiment']} · **Articles:** {v['size']}\n\n"
        if v.get("key_claims"):
            md += "**Key claims:**\n" + "".join([f"- {c}\n" for c in v["key_claims"]])
    return md

def format_recent(recent):
    if not recent:
        return '<p style="font-size:12px;color:#94A3B8;font-style:italic;margin:0;">No recent searches yet.</p>'
    return "".join([f'<div style="font-size:13px;color:#64748B;padding:6px 10px;border-radius:6px;background:#F1F5F9;border:1px solid #E2E8F0;margin-bottom:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{r}</div>' for r in reversed(recent[-6:])])

def analyze_topic(topic, days_back, recent):
    if not topic.strip():
        return "", gr.update(visible=False), None, None, [], format_recent(recent), recent
    articles = fetch_articles(topic.strip(), days_back=int(days_back))
    if not articles:
        return f"No articles found for **'{topic}'**.", gr.update(visible=False), None, None, [], format_recent(recent), recent
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

THEME_JS = """() => {
    const b = document.body;
    const btn = document.querySelector('#tp-dark-btn button');
    if (!btn) return;
    if (b.classList.contains('tp-dark')) { b.classList.remove('tp-dark'); btn.textContent='Dark mode'; }
    else { b.classList.add('tp-dark'); btn.textContent='Light mode'; }
}"""

CSS = """
body { --bg:#F8FAFC;--surf:#FFFFFF;--surf2:#F1F5F9;--bdr:#E2E8F0;--txt:#0F172A;--mid:#374151;--muted:#64748B;--faint:#94A3B8;--acc:#2563EB;--acc-bg:#EFF6FF; }
body.tp-dark { --bg:#0F172A;--surf:#1E293B;--surf2:#334155;--bdr:#334155;--txt:#F1F5F9;--mid:#CBD5E1;--muted:#94A3B8;--faint:#475569;--acc:#60A5FA;--acc-bg:#1E3A5F; }
html,body,.gradio-container,.main,.wrap,.tabitem,.tab-content { background:var(--bg) !important; color:var(--txt) !important; }
.block,.form,.gap,.padded,.container { background:transparent !important; border:none !important; box-shadow:none !important; }
input[type=text],input[type=number],textarea { background:var(--surf) !important; color:var(--txt) !important; border:1.5px solid var(--bdr) !important; border-radius:8px !important; font-size:14px !important; }
input[type=text]:focus,textarea:focus { border-color:var(--acc) !important; box-shadow:0 0 0 3px rgba(37,99,235,0.1) !important; }
label span,.label-wrap span { color:var(--muted) !important; font-size:12.5px !important; font-weight:500 !important; background:transparent !important; border:none !important; padding:0 !important; }
.gradio-container { max-width:1300px !important; margin:0 auto !important; padding:0 !important; font-family:'Inter',sans-serif !important; }
.tab-nav { background:var(--surf) !important; border-bottom:1px solid var(--bdr) !important; padding:0 16px !important; margin:0 !important; }
.tab-nav button { font-size:13px !important; font-weight:600 !important; color:var(--faint) !important; padding:13px 14px !important; background:transparent !important; border:none !important; border-bottom:2px solid transparent !important; }
.tab-nav button.selected { color:var(--acc) !important; border-bottom:2px solid var(--acc) !important; }
#tp-sidebar { background:var(--surf) !important; border-right:1px solid var(--bdr) !important; padding:20px 14px !important; min-height:100vh !important; }
#tp-sidebar button { background:var(--surf2) !important; border:1px solid var(--bdr) !important; color:var(--mid) !important; text-align:left !important; border-radius:8px !important; font-size:13px !important; padding:7px 11px !important; margin-bottom:5px !important; width:100% !important; box-shadow:none !important; justify-content:flex-start !important; font-weight:400 !important; }
#tp-sidebar button:hover { background:var(--acc-bg) !important; color:var(--acc) !important; }
#tp-dark-btn button { background:var(--surf2) !important; border:1px solid var(--bdr) !important; color:var(--mid) !important; border-radius:20px !important; font-size:12px !important; font-weight:600 !important; padding:4px 14px !important; height:28px !important; box-shadow:none !important; width:auto !important; }
#tp-dark-btn button:hover { background:var(--acc-bg) !important; color:var(--acc) !important; }
#tp-main { background:var(--bg) !important; padding:24px 32px !important; }
#tp-chat { background:var(--bg) !important; padding:24px 32px !important; }
#tp-search { background:var(--surf) !important; border:1.5px solid var(--bdr) !important; border-radius:14px !important; padding:20px 22px 18px !important; box-shadow:0 1px 4px rgba(0,0,0,0.05) !important; margin-bottom:18px !important; }
#tp-btn button { background:linear-gradient(135deg,#2563EB,#1D4ED8) !important; border:none !important; border-radius:10px !important; font-weight:600 !important; font-size:14px !important; color:#FFFFFF !important; height:44px !important; width:100% !important; margin-top:10px !important; box-shadow:0 2px 8px rgba(37,99,235,0.28) !important; }
#tp-btn button:hover { background:linear-gradient(135deg,#1D4ED8,#1E40AF) !important; transform:translateY(-1px) !important; }
#tp-results { background:var(--surf) !important; border:1px solid var(--bdr) !important; border-radius:14px !important; padding:24px !important; box-shadow:0 1px 4px rgba(0,0,0,0.04) !important; margin-top:6px !important; }
.prose h2,.md h2 { color:var(--txt) !important; font-size:20px !important; font-weight:700 !important; }
.prose h3,.md h3 { color:var(--txt) !important; font-size:15px !important; font-weight:600 !important; }
.prose p,.md p,.prose li,.md li { color:var(--mid) !important; }
.prose blockquote,.md blockquote { border-left:3px solid var(--acc) !important; background:var(--acc-bg) !important; padding:10px 14px !important; border-radius:0 8px 8px 0 !important; color:var(--acc) !important; }
.gr-dataframe th,thead th { background:var(--surf2) !important; color:var(--muted) !important; font-size:11px !important; font-weight:700 !important; text-transform:uppercase !important; letter-spacing:0.5px !important; }
.gr-dataframe td,tbody td { font-size:13px !important; color:var(--mid) !important; background:var(--surf) !important; border-top:1px solid var(--bdr) !important; }
input[type=range] { accent-color:var(--acc) !important; }
footer { display:none !important; }
"""

HERO = '''<div style="background:linear-gradient(135deg,#EFF6FF 0%,#F8FAFC 55%,#F0FDF4 100%);border:1px solid #E2E8F0;border-radius:14px;padding:24px 28px;margin-bottom:18px;">
<div style="font-size:22px;font-weight:800;color:#0F172A;letter-spacing:-0.5px;margin-bottom:6px;">News Intelligence</div>
<div style="font-size:13.5px;color:#64748B;line-height:1.7;max-width:500px;">Analyse live news for any topic — track narratives, detect sentiment shifts, and surface contradictions across sources.</div>
<div style="display:flex;gap:20px;margin-top:14px;flex-wrap:wrap;">
<div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#475569;"><div style="width:7px;height:7px;border-radius:50%;background:#16A34A;"></div>Narrative clustering</div>
<div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#475569;"><div style="width:7px;height:7px;border-radius:50%;background:#3B82F6;"></div>Sentiment analysis</div>
<div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#475569;"><div style="width:7px;height:7px;border-radius:50%;background:#F59E0B;"></div>Contradiction detection</div>
</div></div>'''

caps = ["Analyse any topic live","Compare two topics","Find contradictions","Summarise coverage","Search stored articles"]
CAP_HTML = "".join([f'<div style="font-size:13px;color:#475569;padding:7px 11px;background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;margin-bottom:5px;">{c}</div>' for c in caps])

with gr.Blocks(title="TrendPulse", css=CSS, theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate", font=["Inter","ui-sans-serif","sans-serif"])) as demo:
    recent_state = gr.State([])
    with gr.Tabs():
        with gr.Tab("Analyse Topic"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=195, elem_id="tp-sidebar"):
                    gr.HTML('<div style="font-size:15px;font-weight:700;color:#0F172A;padding-bottom:14px;border-bottom:1px solid #E2E8F0;margin-bottom:16px;">TrendPulse</div><div style="font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:#94A3B8;margin-bottom:10px;">Try these topics</div>')
                    example_btns = [gr.Button(ex, size="sm") for ex in EXAMPLES]
                    gr.HTML('<div style="font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:#94A3B8;margin:20px 0 10px;">Recent searches</div>')
                    recent_html = gr.HTML('<p style="font-size:12px;color:#94A3B8;font-style:italic;margin:0;">No recent searches yet.</p>')
                    gr.HTML('<div style="height:16px;"></div>')
                    with gr.Row(elem_id="tp-dark-btn"):
                        dark_btn = gr.Button("Dark mode", size="sm")
                with gr.Column(scale=4, elem_id="tp-main"):
                    gr.HTML(HERO)
                    with gr.Group(elem_id="tp-search"):
                        topic_input = gr.Textbox(label="Topic", placeholder="e.g. OpenAI, Indian economy, climate policy ...")
                        days_slider = gr.Slider(minimum=1, maximum=30, value=7, step=1, label="Days back")
                        with gr.Row(elem_id="tp-btn"):
                            analyse_btn = gr.Button("Analyse", variant="primary", size="lg")
                    summary_md = gr.Markdown()
                    with gr.Column(visible=False, elem_id="tp-results") as results_col:
                        with gr.Row():
                            sentiment_plot = gr.Plot(show_label=False)
                            timeline_plot = gr.Plot(show_label=False)
                        gr.HTML('<div style="height:10px"></div>')
                        articles_table = gr.Dataframe(headers=["Title","Source","Date","Cluster"], label="All articles", wrap=True, row_count=8)
            for btn in example_btns:
                btn.click(fn=lambda x=btn.value: x, outputs=topic_input)
            dark_btn.click(fn=None, js=THEME_JS)
            analyse_btn.click(fn=analyze_topic, inputs=[topic_input, days_slider, recent_state], outputs=[summary_md, results_col, sentiment_plot, timeline_plot, articles_table, recent_html, recent_state])

        with gr.Tab("Chat with Agent"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=195, elem_id="tp-sidebar"):
                    gr.HTML(f'<div style="font-size:15px;font-weight:700;color:#0F172A;padding-bottom:14px;border-bottom:1px solid #E2E8F0;margin-bottom:16px;">TrendPulse</div><div style="font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:#94A3B8;margin-bottom:12px;">What the agent can do</div>{CAP_HTML}<div style="font-size:10px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:#94A3B8;margin:20px 0 10px;">Example prompts</div><div style="font-size:12px;color:#94A3B8;line-height:2.3;font-style:italic;">"Narratives around AI regulation?"<br>"Compare OpenAI vs DeepMind"<br>"Contradictions in US economy?"<br>"Summarise Fed rate cut news"</div>')
                with gr.Column(scale=4, elem_id="tp-chat"):
                    gr.ChatInterface(fn=agent_chat, title="")

if __name__ == "__main__":
    demo.launch(show_api=False)
