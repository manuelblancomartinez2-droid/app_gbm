"""
app_gbm.py — Interfaz (Streamlit + Plotly). La matemática vive en motor_gbm.py
Ejecutar:  streamlit run app_gbm.py
Requisitos: pip install streamlit plotly yfinance scipy numpy pandas
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import motor_gbm
from motor_gbm import BONOS13W, STEPS, T

st.set_page_config(page_title="Monte Carlo · GBM multiactivo", page_icon="◆",
                   layout="wide", initial_sidebar_state="expanded")

# ════════════════════════════════════════════════════════════════════════
# Estilo
# ════════════════════════════════════════════════════════════════════════
BG, PANEL, BORDER, GRID = "#0B0F17", "#111827", "#1F2937", "#18202E"
TXT, MUTED = "#E6EAF2", "#8A94A6"
C_N, C_T = "#4F8CFF", "#F5A524"            # Normal / t-Student
C_BS = "#A78BFA"                            # Black-Scholes
C_POS, C_NEG = "#22C55E", "#EF4444"
PALETA = ["#4F8CFF", "#A78BFA", "#22D3EE", "#34D399", "#F5A524",
          "#F472B6", "#FB7185", "#FACC15"]


def rgba(h, a):
    h = h.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stApp {{ font-family:'Inter',sans-serif; }}
.stApp {{ background:{BG}; color:{TXT}; }}
header[data-testid="stHeader"] {{ background:transparent; }}
section[data-testid="stSidebar"] {{ background:{PANEL}; border-right:1px solid {BORDER}; }}
section[data-testid="stSidebar"] * {{ color:{TXT}; }}
.block-container {{ padding-top:1.4rem; max-width:1500px; }}
h1,h2,h3,h4 {{ color:{TXT}; letter-spacing:-.01em; }}
p, label, span, div {{ color:inherit; }}
.hdr {{ display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; flex-wrap:wrap; }}
.hdr-title {{ font-size:1.6rem; font-weight:700; margin:0; color:{TXT}; }}
.hdr-sub {{ color:{MUTED}; font-size:.88rem; margin-top:.15rem; }}
.badge {{ font-family:'JetBrains Mono',monospace; font-size:.75rem; color:{MUTED};
          border:1px solid {BORDER}; border-radius:999px; padding:.2rem .7rem; }}
.kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.7rem; margin:1rem 0 .6rem; }}
.kpi {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:12px; padding:.75rem 1rem; }}
.kpi .l {{ color:{MUTED}; font-size:.68rem; text-transform:uppercase; letter-spacing:.09em; }}
.kpi .v {{ font-family:'JetBrains Mono',monospace; font-size:1.2rem; font-weight:500; margin-top:.2rem; color:{TXT}; }}
.kpi .s {{ color:{MUTED}; font-size:.72rem; margin-top:.1rem; }}
.sec {{ color:{MUTED}; font-size:.72rem; text-transform:uppercase; letter-spacing:.11em;
        margin:1.1rem 0 .35rem; border-bottom:1px solid {BORDER}; padding-bottom:.3rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap:.2rem; border-bottom:1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{ background:transparent; color:{MUTED}; padding:.55rem 1rem; }}
.stTabs [aria-selected="true"] {{ color:{TXT}; }}
div[data-testid="stDataFrame"] {{ border:1px solid {BORDER}; border-radius:10px; }}
.katex {{ color:{TXT}; font-size:1.0em; }}
</style>""", unsafe_allow_html=True)


def estilo(fig, h=420, legend=True, top=36):
    fig.update_layout(
        template="plotly_dark", height=h,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TXT, size=12),
        margin=dict(l=8, r=8, t=top, b=8), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=BORDER,
                        font=dict(family="JetBrains Mono, monospace", color=TXT)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=BORDER, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=BORDER, tickfont=dict(color=MUTED))
    fig.update_annotations(font=dict(color=MUTED, size=12))
    return fig


def sec(t):
    st.markdown(f'<div class="sec">{t}</div>', unsafe_allow_html=True)



# ════════════════════════════════════════════════════════════════════════
# Puente con el motor (caché 1 h en memoria, sin copiar los arrays)
# ════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False, max_entries=2, ttl=3600)
def motor(tickers: tuple, N_sims: int, nu: int):
    return motor_gbm.simular(tickers, N_sims, nu)


# ════════════════════════════════════════════════════════════════════════
# Gráficos
# ════════════════════════════════════════════════════════════════════════
def cono(fig, R, i, color, row=None, col=None, legend=True):
    x = np.arange(R["media"].shape[0])
    kw = dict(row=row, col=col) if row else {}
    g = dict(legendgroup="cono", showlegend=legend)
    fig.add_trace(go.Scatter(x=x, y=R["p95"][:, i], line=dict(width=0), hoverinfo="skip",
                             showlegend=False, legendgroup="cono"), **kw)
    fig.add_trace(go.Scatter(x=x, y=R["p5"][:, i], fill="tonexty", fillcolor=rgba(color, .12),
                             line=dict(width=0), name="P5–P95", hoverinfo="skip", **g), **kw)
    fig.add_trace(go.Scatter(x=x, y=R["p75"][:, i], line=dict(width=0), hoverinfo="skip",
                             showlegend=False, legendgroup="cono"), **kw)
    fig.add_trace(go.Scatter(x=x, y=R["p25"][:, i], fill="tonexty", fillcolor=rgba(color, .26),
                             line=dict(width=0), name="P25–P75", hoverinfo="skip", **g), **kw)
    fig.add_trace(go.Scatter(x=x, y=R["Mediana"][:, i], line=dict(color=color, width=2),
                             name="Mediana", **g), **kw)
    fig.add_trace(go.Scatter(x=x, y=R["media"][:, i], line=dict(color=TXT, width=1.2, dash="dot"),
                             name="Media", **g), **kw)


@st.cache_resource(show_spinner=False, max_entries=16)
def fig_animacion(tickers, N_sims, nu, i, n_paths=80, stride=4, n_bins=70):
    R = motor(tickers, N_sims, nu)
    P, Pt = R["precios"][:, :, i], R["precios_t"][:, :, i]
    steps = P.shape[1] - 1
    So = R["So"][i]
    color = PALETA[i % len(PALETA)]

    lo = min(np.percentile(P[:, -1], .5), np.percentile(Pt[:, -1], .5))
    hi = max(np.percentile(P[:, -1], 99.5), np.percentile(Pt[:, -1], 99.5))
    pad = (hi - lo) * .04
    bins = np.linspace(lo - pad, hi + pad, n_bins + 1)
    centros, ancho = (bins[:-1] + bins[1:]) / 2, bins[1] - bins[0]

    def dens(x):
        c, _ = np.histogram(x, bins=bins)
        return c / len(x)

    idx = list(range(0, steps + 1, stride))
    if idx[-1] != steps:
        idx.append(steps)
    muestra = P[:n_paths]
    q = {k: R[k][:, i] for k in ("p5", "p25", "p75", "p95", "Mediana", "media")}

    def datos(k):
        x = np.arange(k + 1)
        xp = np.tile(np.append(x, np.nan), n_paths)
        yp = np.hstack([muestra[:, :k + 1], np.full((n_paths, 1), np.nan)]).ravel()
        return [go.Scatter(x=xp, y=yp), go.Scatter(x=x, y=q["p95"][:k + 1]),
                go.Scatter(x=x, y=q["p5"][:k + 1]), go.Scatter(x=x, y=q["p75"][:k + 1]),
                go.Scatter(x=x, y=q["p25"][:k + 1]), go.Scatter(x=x, y=q["Mediana"][:k + 1]),
                go.Scatter(x=x, y=q["media"][:k + 1]),
                go.Bar(x=dens(P[:, k]), y=centros), go.Scatter(x=dens(Pt[:, k]), y=centros)]

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[.74, .26],
                        horizontal_spacing=.035)
    d = datos(steps)
    estilos = [
        dict(mode="lines", line=dict(color=rgba(color, .16), width=1), name=f"{n_paths} trayectorias",
             hoverinfo="skip"),
        dict(line=dict(width=0), hoverinfo="skip", showlegend=False),
        dict(fill="tonexty", fillcolor=rgba(color, .13), line=dict(width=0), name="P5–P95", hoverinfo="skip"),
        dict(line=dict(width=0), hoverinfo="skip", showlegend=False),
        dict(fill="tonexty", fillcolor=rgba(color, .28), line=dict(width=0), name="P25–P75", hoverinfo="skip"),
        dict(line=dict(color=color, width=2.4), name="Mediana"),
        dict(line=dict(color=TXT, width=1.3, dash="dot"), name="Media"),
        dict(orientation="h", width=ancho, marker=dict(color=rgba(C_N, .55), line=dict(width=0)),
             name="S<sub>t</sub> Normal", hovertemplate="S=%{y:.2f}<br>p=%{x:.3%}<extra>Normal</extra>"),
        dict(mode="lines", line=dict(color=C_T, width=1.6, shape="vhv"),
             name=f"S<sub>t</sub> t-Student ν={nu}",
             hovertemplate="S=%{y:.2f}<br>p=%{x:.3%}<extra>t-Student</extra>"),
    ]
    for j, (tr, s) in enumerate(zip(d, estilos)):
        tr.update(**s)
        fig.add_trace(tr, row=1, col=1 if j < 7 else 2)

    frames = [go.Frame(data=datos(k), traces=list(range(9)), name=str(k)) for k in idx]
    fig.frames = frames
    xmax = max(max(dens(P[:, k]).max(), dens(Pt[:, k]).max()) for k in idx if k >= steps * .25) * 1.12

    fig.add_hline(y=So, line=dict(color=MUTED, width=1, dash="dash"), row=1, col=2)
    fig.add_hline(y=So, line=dict(color=MUTED, width=1, dash="dash"), row=1, col=1,
                  annotation_text=f"S₀ {So:,.2f}", annotation_position="top left",
                  annotation_font=dict(color=MUTED, size=11))
    fig.update_xaxes(range=[0, steps], title_text="Día hábil", row=1, col=1)
    fig.update_xaxes(range=[0, xmax], tickformat=".1%", title_text="Densidad en t", row=1, col=2)
    fig.update_yaxes(range=[bins[0], bins[-1]], title_text="Precio", row=1, col=1)
    estilo(fig, h=560)
    fig.update_layout(
        barmode="overlay", bargap=0, margin=dict(l=8, r=8, t=40, b=110),
        updatemenus=[dict(
            type="buttons", direction="left", showactive=False, x=0, y=-.13, xanchor="left", yanchor="top",
            bgcolor=PANEL, bordercolor=BORDER, font=dict(color=TXT), pad=dict(r=6, t=0),
            buttons=[
                dict(label="▶ Desde t=0", method="animate",
                     args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=False,
                                      transition=dict(duration=0), mode="immediate")]),
                dict(label="❚❚ Pausa", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                dict(label="▶▶ Continuar", method="animate",
                     args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True,
                                      transition=dict(duration=0), mode="immediate")]),
            ])],
        sliders=[dict(
            active=len(idx) - 1, x=.27, len=.73, y=-.1, yanchor="top",
            bgcolor=BORDER, activebgcolor=color, bordercolor=BORDER, tickcolor=MUTED,
            font=dict(color=MUTED, size=10),
            currentvalue=dict(prefix="Día ", font=dict(color=TXT, size=13), xanchor="right"),
            steps=[dict(method="animate", label=str(k),
                        args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                             transition=dict(duration=0))]) for k in idx])],
    )
    return fig


def hist_overlay(a, b, nombre_a, nombre_b, n_bins=80, clip=(.1, 99.9)):
    lo = min(np.percentile(a, clip[0]), np.percentile(b, clip[0]))
    hi = max(np.percentile(a, clip[1]), np.percentile(b, clip[1]))
    bins = np.linspace(lo, hi, n_bins + 1)
    c = (bins[:-1] + bins[1:]) / 2
    fig = go.Figure()
    ha, _ = np.histogram(a, bins=bins)
    hb, _ = np.histogram(b, bins=bins)
    fig.add_trace(go.Bar(x=c, y=ha / len(a), width=bins[1] - bins[0], name=nombre_a,
                         marker=dict(color=rgba(C_N, .55), line=dict(width=0)),
                         hovertemplate="%{x:.2f}<br>p=%{y:.3%}<extra>" + nombre_a + "</extra>"))
    fig.add_trace(go.Scatter(x=c, y=hb / len(b), name=nombre_b, mode="lines",
                             line=dict(color=C_T, width=1.8, shape="hvh"),
                             hovertemplate="%{x:.2f}<br>p=%{y:.3%}<extra>" + nombre_b + "</extra>"))
    fig.update_layout(barmode="overlay", bargap=0)
    fig.update_yaxes(tickformat=".1%")
    return fig


def vline(fig, x, color, texto, dash="solid", nivel=0):
    fig.add_vline(x=x, line=dict(color=color, width=1.4, dash=dash),
                  annotation_text=texto, annotation_position="top right",
                  annotation=dict(yshift=-16 * nivel),
                  annotation_font=dict(color=color, size=11))


# ════════════════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ◆ Parámetros")
    with st.form("params"):
        txt = st.text_input("Tickers", "SPY, QQQ, AAPL, MSFT, GLD")
        N_sims = st.select_slider("Nº simulaciones", [1000, 2500, 5000, 10000], value=10000)
        nu = st.slider("Grados de libertad ν (t-Student)", 3, 30, 5)
        st.form_submit_button("Simular", width="stretch", type="primary")
    st.markdown(f"""<div style="color:{MUTED};font-size:.78rem;line-height:1.6;margin-top:.6rem">
    Histórico: 1 año diario (yfinance)<br>Pasos: {STEPS} · T = {T:.0f} año · seed = 42<br>
    Tipo libre de riesgo: {BONOS13W}<br>Strike: K = S₀ (ATM)</div>""", unsafe_allow_html=True)

TICKERS = tuple(dict.fromkeys(t.strip().upper() for t in txt.split(",") if t.strip()))
if len(TICKERS) < 2:
    st.error("Introduce al menos 2 tickers (la matriz de correlación/Cholesky lo requiere).")
    st.stop()

try:
    with st.spinner(f"Descargando datos y simulando {N_sims:,} × {STEPS} × {len(TICKERS)}…"):
        R = motor(TICKERS, N_sims, nu)
except KeyError as e:
    st.error(f"Ticker sin datos en yfinance: {e}")
    st.stop()
except np.linalg.LinAlgError:
    st.error("La matriz de correlación no es definida positiva (¿tickers duplicados o colineales?).")
    st.stop()

n = len(TICKERS)
So, r, sigma = R["So"], R["r"], R["sigma"]
COL = {t: PALETA[k % len(PALETA)] for k, t in enumerate(TICKERS)}

# ════════════════════════════════════════════════════════════════════════
# Cabecera
# ════════════════════════════════════════════════════════════════════════
ult = R["Close"].index[-1]
st.markdown(f"""<div class="hdr"><div>
<div class="hdr-title">Monte Carlo · GBM multiactivo correlacionado</div>
<div class="hdr-sub">Normal vs t-Student · VaR / CVaR · Black-Scholes vs Monte Carlo · Validación martingala</div>
</div><span class="badge">último cierre {pd.Timestamp(ult).strftime('%d %b %Y')}</span></div>""",
            unsafe_allow_html=True)

kpis = [("Tipo libre de riesgo", f"{r:.3%}", "^IRX · 13 semanas"),
        ("Simulaciones", f"{N_sims:,}", f"{STEPS} pasos · dt = 1/{STEPS}"),
        ("Horizonte", f"{T:.0f} año", "T"),
        ("Colas pesadas", f"ν = {nu}", "t-Student reescalada"),
        ("Activos", f"{n}", " · ".join(TICKERS)),
        ("Observaciones", f"{len(R['retornosLog'])}", "log-retornos diarios")]
st.markdown('<div class="kpi-grid">' + "".join(
    f'<div class="kpi"><div class="l">{l}</div><div class="v">{v}</div><div class="s">{s}</div></div>'
    for l, v, s in kpis) + "</div>", unsafe_allow_html=True)

tabs = st.tabs(["Resumen", "Evolución temporal", "Riesgo · VaR / CVaR", "Opciones",
                "Correlación", "Validación martingala"])

# ── Resumen ─────────────────────────────────────────────────────────────
with tabs[0]:
    sec("Resultados por activo")
    df = pd.DataFrame({
        "Activo": TICKERS, "S₀": So, "σ anual": sigma * 100,
        "E[S_T]": R["media"][-1], "Mediana S_T": R["Mediana"][-1],
        "P5 S_T": R["p5"][-1], "P95 S_T": R["p95"][-1],
        "VaR95 N": R["var95"], "CVaR95 N": R["Cvar"],
        "VaR95 t": R["var95_t"], "CVaR95 t": R["Cvar_t"],
        "Call BS": R["bs_call"], "Call MC N": R["mc_call_norm"], "Call MC t": R["mc_call_stud"],
        "Err. martingala": R["errores"],
    })
    cfg = {c: st.column_config.NumberColumn(format="%.2f") for c in df.columns[1:]}
    cfg["σ anual"] = st.column_config.NumberColumn(format="%.2f%%")
    cfg["Err. martingala"] = st.column_config.NumberColumn(format="%.3f%%")
    st.dataframe(df, hide_index=True, width="stretch", column_config=cfg)

    sec("Conos de percentiles · GBM Normal")
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=TICKERS,
                        vertical_spacing=.12, horizontal_spacing=.05)
    for k, t in enumerate(TICKERS):
        rr, cc = k // cols + 1, k % cols + 1
        cono(fig, R, k, COL[t], row=rr, col=cc, legend=(k == 0))
        fig.add_hline(y=So[k], line=dict(color=MUTED, width=1, dash="dash"), row=rr, col=cc)
    estilo(fig, h=300 * rows, top=60)
    st.plotly_chart(fig, width="stretch")

# ── Evolución temporal ──────────────────────────────────────────────────
with tabs[1]:
    activo = st.radio("Activo", TICKERS, horizontal=True, key="act_evo")
    i = TICKERS.index(activo)

    sec("Reproducción de la simulación")
    st.caption("Trayectorias y cono crecen día a día; a la derecha, la distribución transversal "
               "de S_t (Normal vs t-Student) en ese instante.")
    st.plotly_chart(fig_animacion(TICKERS, N_sims, nu, i), width="stretch")

    sec("Instantánea en un día concreto")
    t = st.slider("Día hábil", 0, STEPS, STEPS // 2, key="t_snap")
    m = [("Media", R["media"][t, i]), ("Mediana", R["Mediana"][t, i]),
         ("P5", R["p5"][t, i]), ("P25", R["p25"][t, i]),
         ("P75", R["p75"][t, i]), ("P95", R["p95"][t, i])]
    st.markdown('<div class="kpi-grid">' + "".join(
        f'<div class="kpi"><div class="l">{l} · día {t}</div><div class="v">{v:,.2f}</div>'
        f'<div class="s" style="color:{C_POS if v >= So[i] else C_NEG}">{(v / So[i] - 1):+.2%} vs S₀</div></div>'
        for l, v in m) + "</div>", unsafe_allow_html=True)
    if t == 0:
        st.info("En t = 0 todas las trayectorias valen S₀.")
    else:
        fig = hist_overlay(R["precios"][:, t, i], R["precios_t"][:, t, i], "Normal", f"t-Student ν={nu}")
        vline(fig, So[i], MUTED, "S₀", "dash")
        vline(fig, R["p5"][t, i], C_NEG, "P5", "dot", 1)
        vline(fig, R["Mediana"][t, i], C_N, "Mediana")
        vline(fig, R["p95"][t, i], C_POS, "P95", "dot", 1)
        fig.update_xaxes(title_text=f"S_t · día {t}")
        st.plotly_chart(estilo(fig, h=380), width="stretch")

# ── Riesgo ──────────────────────────────────────────────────────────────
with tabs[2]:
    st.latex(r"L = S_0 - S_T,\qquad \mathrm{VaR}_{95} = Q_{0.95}(L),\qquad "
             r"\mathrm{CVaR}_{95} = \mathbb{E}\left[L \mid L \ge \mathrm{VaR}_{95}\right]")
    pct = st.toggle("Expresar en % de S₀", value=True)
    esc = 100 / So if pct else np.ones(n)
    unidad = "%" if pct else "$"

    sec("VaR y CVaR al 95% · Normal vs t-Student")
    fig = go.Figure()
    for nom, v, c in (("VaR Normal", R["var95"], C_N), ("CVaR Normal", R["Cvar"], rgba(C_N, .45)),
                      (f"VaR t (ν={nu})", R["var95_t"], C_T), (f"CVaR t (ν={nu})", R["Cvar_t"], rgba(C_T, .45))):
        fig.add_trace(go.Bar(x=TICKERS, y=v * esc, name=nom, marker_color=c,
                             text=[f"{x:.1f}" for x in v * esc], textposition="outside",
                             textfont=dict(color=MUTED, size=10),
                             hovertemplate="%{x}: %{y:.2f}" + unidad + "<extra>" + nom + "</extra>"))
    fig.update_layout(barmode="group", bargap=.25, bargroupgap=.06)
    fig.update_yaxes(title_text=f"Pérdida ({unidad})")
    st.plotly_chart(estilo(fig, h=400), width="stretch")

    sec("Distribución de pérdidas a vencimiento")
    activo = st.radio("Activo", TICKERS, horizontal=True, key="act_risk")
    i = TICKERS.index(activo)
    fig = hist_overlay(R["perdidas_norm"][:, i], R["perdidas_stud"][:, i], "Normal", f"t-Student ν={nu}")
    vline(fig, R["var95"][i], C_N, "VaR N")
    vline(fig, R["Cvar"][i], C_N, "CVaR N", "dot", 1)
    vline(fig, R["var95_t"][i], C_T, "VaR t", "solid", 2)
    vline(fig, R["Cvar_t"][i], C_T, "CVaR t", "dot", 3)
    fig.update_xaxes(title_text="Pérdida L = S₀ − S_T ($)")
    st.plotly_chart(estilo(fig, h=380), width="stretch")
    st.caption("Eje recortado a los percentiles 0.1–99.9 para legibilidad; VaR/CVaR calculados sobre la muestra completa.")

# ── Opciones ────────────────────────────────────────────────────────────
with tabs[3]:
    st.latex(r"C_{BS} = S_0\,N(d_1) - K e^{-rT} N(d_2),\qquad "
             r"C_{MC} = e^{-rT}\,\frac{1}{N}\sum_{j=1}^{N}\max(S_T^{(j)}-K,\,0),\qquad K = S_0")
    sec("Call ATM · Black-Scholes vs Monte Carlo")
    fig = go.Figure()
    for nom, v, c in (("Black-Scholes", R["bs_call"], C_BS), ("MC Normal", R["mc_call_norm"], C_N),
                      (f"MC t (ν={nu})", R["mc_call_stud"], C_T)):
        fig.add_trace(go.Bar(x=TICKERS, y=v, name=nom, marker_color=c,
                             text=[f"{x:.2f}" for x in v], textposition="outside",
                             textfont=dict(color=MUTED, size=10),
                             hovertemplate="%{x}: %{y:.4f}<extra>" + nom + "</extra>"))
    fig.update_layout(barmode="group", bargap=.25, bargroupgap=.06)
    fig.update_yaxes(title_text="Prima ($)")
    st.plotly_chart(estilo(fig, h=400), width="stretch")

    sec("Detalle")
    dfo = pd.DataFrame({
        "Activo": TICKERS, "S₀ = K": So, "σ": sigma * 100,
        "Black-Scholes": R["bs_call"], "MC Normal": R["mc_call_norm"], "MC t-Student": R["mc_call_stud"],
        "Δ MC N vs BS": (R["mc_call_norm"] / R["bs_call"] - 1) * 100,
        "Δ MC t vs BS": (R["mc_call_stud"] / R["bs_call"] - 1) * 100,
    })
    cfg = {c: st.column_config.NumberColumn(format="%.4f") for c in dfo.columns[1:]}
    cfg["S₀ = K"] = st.column_config.NumberColumn(format="%.2f")
    cfg["σ"] = st.column_config.NumberColumn(format="%.2f%%")
    for c in ("Δ MC N vs BS", "Δ MC t vs BS"):
        cfg[c] = st.column_config.NumberColumn(format="%+.3f%%")
    st.dataframe(dfo, hide_index=True, width="stretch", column_config=cfg)

# ── Correlación ─────────────────────────────────────────────────────────
with tabs[4]:
    st.latex(r"\rho = \mathrm{corr}(\ln S_t - \ln S_{t-1}),\qquad \rho = LL^{\top},\qquad "
             r"W = Z L^{\top},\qquad \sigma = \mathrm{sd}(r_{\log})\sqrt{252}")
    escala = [[0, "#EF4444"], [.5, PANEL], [1, C_N]]
    c1, c2 = st.columns(2)
    with c1:
        sec("Matriz de correlación ρ")
        fig = go.Figure(go.Heatmap(z=R["corr"], x=TICKERS, y=TICKERS, zmin=-1, zmax=1, colorscale=escala,
                                   text=np.round(R["corr"], 2), texttemplate="%{text}",
                                   textfont=dict(color=TXT), xgap=2, ygap=2,
                                   hovertemplate="%{y} · %{x}: %{z:.3f}<extra></extra>",
                                   colorbar=dict(thickness=10, tickfont=dict(color=MUTED))))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(estilo(fig, h=420, legend=False), width="stretch")
    with c2:
        sec("Factor de Cholesky L")
        Lm = np.where(np.tril(np.ones_like(R["L"])) > 0, R["L"], np.nan)
        fig = go.Figure(go.Heatmap(z=Lm, x=TICKERS, y=TICKERS, zmin=-1, zmax=1, colorscale=escala,
                                   text=np.where(np.isnan(Lm), "", np.round(np.nan_to_num(Lm), 3).astype(str)),
                                   texttemplate="%{text}", textfont=dict(color=TXT), xgap=2, ygap=2,
                                   hovertemplate="L[%{y}, %{x}] = %{z:.4f}<extra></extra>",
                                   colorbar=dict(thickness=10, tickfont=dict(color=MUTED))))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(estilo(fig, h=420, legend=False), width="stretch")

    sec("Volatilidad anualizada histórica")
    fig = go.Figure(go.Bar(x=TICKERS, y=sigma * 100, marker_color=[COL[t] for t in TICKERS],
                           text=[f"{s:.1%}" for s in sigma], textposition="outside",
                           textfont=dict(color=MUTED), hovertemplate="%{x}: %{y:.2f}%<extra></extra>"))
    fig.update_yaxes(title_text="σ (%)")
    st.plotly_chart(estilo(fig, h=320, legend=False), width="stretch")

# ── Validación martingala ───────────────────────────────────────────────
with tabs[5]:
    st.latex(r"e^{-rT}\,\mathbb{E}^{\mathbb{Q}}[S_T] = S_0 \quad\Rightarrow\quad "
             r"\varepsilon = \frac{\left|e^{-rT}\,\overline{S_T} - S_0\right|}{S_0}\times 100")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        sec("Error relativo (%)")
        fig = go.Figure(go.Bar(x=TICKERS, y=R["errores"], marker_color=[COL[t] for t in TICKERS],
                               text=[f"{e:.3f}%" for e in R["errores"]], textposition="outside",
                               textfont=dict(color=MUTED), hovertemplate="%{x}: %{y:.4f}%<extra></extra>"))
        fig.update_yaxes(title_text="ε (%)")
        st.plotly_chart(estilo(fig, h=360, legend=False), width="stretch")
    with c2:
        sec("Detalle")
        dfm = pd.DataFrame({"Activo": TICKERS, "S₀": So, "e^{-rT}·E[S_T]": R["martingala"],
                            "ε (%)": R["errores"]})
        st.dataframe(dfm, hide_index=True, width="stretch", column_config={
            "S₀": st.column_config.NumberColumn(format="%.4f"),
            "e^{-rT}·E[S_T]": st.column_config.NumberColumn(format="%.4f"),
            "ε (%)": st.column_config.NumberColumn(format="%.4f%%")})
