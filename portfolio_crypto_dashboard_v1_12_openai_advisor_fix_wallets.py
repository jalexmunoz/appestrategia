# portfolio_crypto_dashboard_v1_12_openai_advisor_fix_wallets.py
import os, textwrap, re, requests
import pandas as pd
import numpy as np
import streamlit as st

# ================= ENV =================
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

st.set_page_config(page_title="Crypto Dashboard — v1.12 · OpenAI Advisor", layout="wide")

# ================= THEME =================
st.markdown("""
<style>
:root, .stApp { --bg:#0B1220; --text:#e5e7eb; --muted:#94a3b8; --chip:rgba(255,255,255,.07); --green:#10b981; --red:#ef4444; --amber:#f59e0b }
.stApp { background:var(--bg); color:var(--text) }
[data-testid="stSidebar"]{ width:280px; min-width:280px }
.title{ font-size:1.9rem; font-weight:800 }
.kpi{ background:linear-gradient(180deg, rgba(99,102,241,.14), rgba(99,102,241,.05)); border:1px solid rgba(99,102,241,.28); padding:12px 14px; border-radius:14px }
.subtle{ color:var(--muted) }
.chip{ display:inline-block; padding:4px 10px; border-radius:999px; font-size:.78rem; background:var(--chip); border:1px solid rgba(255,255,255,.08); margin-right:6px; font-weight:600 }
.chip.win{ color:var(--green) } .chip.lose{ color:var(--red) } .chip.flat{ color:var(--amber) }
.chip.tgt{ color:#c084fc } .chip.stop{ color:#fca5a5 } .chip.err{ color:#f97316 }
.chip.trend{ color:#22d3ee } .chip.unusual{ color:#60a5fa }
[data-testid="stDataFrame"] div[role="gridcell"], [data-testid="stDataFrame"] div[role="columnheader"] { color: var(--text) !important; }
label[data-testid="stWidgetLabel"] p { color:#e5e7eb!important; white-space:normal!important; overflow:visible!important; text-overflow:unset!important; }
.stTextInput input, [data-testid="stNumberInput"] input, .stTextArea textarea {
  color:#e5e7eb!important; background-color:rgba(255,255,255,0.06)!important; border:1px solid rgba(255,255,255,0.18)!important; caret-color:#e5e7eb!important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🟣 Crypto Dashboard — v1.12 · OpenAI Advisor</div>', unsafe_allow_html=True)
st.caption("Base: CMC quotes (precios), Fear & Greed, Trending CoinGecko, heurísticos. + Panel con OpenAI.")

# ================= SIDEBAR =================
CMC_KEY    = os.getenv("CMC_API_KEY","")
OPENAI_KEY = os.getenv("OPENAI_API_KEY","")
c1, c2 = st.sidebar.columns([2,1])
with c1: st.text_input("CMC API Key (opcional, solo precios)", value=CMC_KEY, type="password", key="cmc_key")
with c2: st.toggle("Auto-refresh", value=True)
st.sidebar.text_input("OPENAI_API_KEY", value=OPENAI_KEY, type="password", key="openai_key")
cache_min  = st.sidebar.number_input("Cache precios (min)", 0, 120, 5)
st.sidebar.slider("Frecuencia refresh (min)", 1, 30, 5)
band       = st.sidebar.slider("Cerca Target/Stop (±%)", 1.0, 20.0, 5.0, .5)
st.sidebar.markdown("---")
mover_thr   = st.sidebar.slider("Umbral Top Movers (|24h%| ≥)", 0.0, 20.0, 5.0, 0.5)
unusual_thr = st.sidebar.slider("Umbral 'Unusual' (|24h%| ≥)", 2.0, 20.0, 8.0, 0.5)

# ================= CSV =================
up = st.file_uploader("Upload crypto CSV", type=["csv"])
if up is None:
    st.info("CSV mínimo: Symbol, Quantity, Total Cost (USD), Avg Cost (USD). Opcional: Price (USD), Target Price, Stop Price, Wallet, Note.")
    st.stop()
df = pd.read_csv(up)
df.columns = [c.strip() for c in df.columns]
if "Symbol" not in df.columns:
    st.error("El CSV debe tener columna 'Symbol'."); st.stop()
df["Symbol"] = df["Symbol"].astype(str).str.upper()

# -------- LIMPIADOR ROBUSTO DE MONEDA --------
def clean_money(x):
    """Convierte '$4,613.88' -> 4613.88; maneja '', None, 'nan', '—'."""
    if x is None: return np.nan
    if isinstance(x, (int, float, np.number)): return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan","none","null","—","-"}: return np.nan
    s = re.sub(r"[^\d\.\-]", "", s)
    try: return float(s)
    except: return np.nan

# Asegura columnas base
for c in ["Quantity","Total Cost (USD)","Avg Cost (USD)","Price (USD)","Target Price","Stop Price","Wallet"]:
    if c not in df.columns: df[c] = np.nan
for c in ["Quantity","Total Cost (USD)","Avg Cost (USD)","Price (USD)","Target Price","Stop Price"]:
    df[c] = df[c].apply(clean_money)
df["Wallet"] = df["Wallet"].fillna("")

# ================= HELPERS =================
def safe_float(x, default=np.nan):
    try: return float(x)
    except Exception: return default

# ================= CMC QUOTES =================
@st.cache_data(ttl=60*5, show_spinner=False)
def cmc_quotes_raw(symbols, cmc_key):
    url = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest"
    headers = {"X-CMC_PRO_API_KEY": cmc_key}
    r = requests.get(url, headers=headers, params={"symbol":",".join(symbols),"convert":"USD"}, timeout=30)
    r.raise_for_status()
    return r.json()

def get_quotes(symbols, cmc_key):
    out = {"quotes":{}, "pct24h":{}, "pct7d":{}}
    if not cmc_key: return out
    try:
        j = cmc_quotes_raw(symbols, cmc_key)
        d = j.get("data",{})
        for s, arr in d.items():
            it = arr[0] if isinstance(arr, list) and arr else arr
            q  = it.get("quote",{}).get("USD",{})
            sU = s.upper()
            out["quotes"][sU] = safe_float(q.get("price"))
            out["pct24h"][sU] = safe_float(q.get("percent_change_24h"), 0.0)
            out["pct7d"][sU]  = safe_float(q.get("percent_change_7d"), 0.0)
    except Exception as e:
        st.warning(f"No se pudieron traer precios de CMC (usando Price del CSV): {e}")
    return out

symbols = df["Symbol"].dropna().unique().tolist()
quotes, pct24h, pct7d = {}, {}, {}
if st.session_state.cmc_key:
    cmc = get_quotes(symbols, st.session_state.cmc_key)
    quotes, pct24h, pct7d = cmc["quotes"], cmc["pct24h"], cmc["pct7d"]
    if quotes: st.caption(f"💹 Precios live desde CMC (cache {cache_min} min).")

# ================= FG / TRENDING =================
@st.cache_data(ttl=60*30, show_spinner=False)
def altme_fng():
    r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=20); r.raise_for_status()
    d = r.json().get("data",[])
    if not d: return None
    it = d[0]
    return {"value": int(it.get("value")), "classification": it.get("value_classification")}

@st.cache_data(ttl=60*10, show_spinner=False)
def coingecko_trending_symbols():
    r = requests.get("https://api.coingecko.com/api/v3/search/trending", timeout=20); r.raise_for_status()
    data = r.json().get("coins",[])
    return { (it.get("item",{}).get("symbol") or "").upper() for it in data if it.get("item") }

# ================= METRICS (ANTES DEL SPLIT) =================
df["Live Price (USD)"] = df["Symbol"].map(quotes) if quotes else np.nan
df["Live Price (USD)"] = df["Live Price (USD)"].apply(clean_money)

# Fallback: live → price CSV → avg cost
df["Used Price (USD)"] = df["Live Price (USD)"]
mask = df["Used Price (USD)"].isna() | (df["Used Price (USD)"] <= 0)
df.loc[mask, "Used Price (USD)"] = df.loc[mask, "Price (USD)"]
mask = df["Used Price (USD)"].isna() | (df["Used Price (USD)"] <= 0)
df.loc[mask, "Used Price (USD)"] = df.loc[mask, "Avg Cost (USD)"]

df["Quantity"]            = df["Quantity"].fillna(0.0)
df["Total Cost (USD)"]    = df["Total Cost (USD)"].fillna(0.0)
df["Current Value (USD)"] = (df["Quantity"] * df["Used Price (USD)"]).fillna(0.0)
df["P&L (USD)"]           = df["Current Value (USD)"] - df["Total Cost (USD)"]
df["P&L %"]               = np.where(df["Total Cost (USD)"]>0, 100*df["P&L (USD)"]/df["Total Cost (USD)"], 0.0)

total_val = float(df["Current Value (USD)"].sum(skipna=True))
df["% Portfolio"] = np.where(total_val>0, 100*df["Current Value (USD)"]/total_val, 0.0)

# % cambio (si no vienen, NaN)
if "24h %" not in df.columns: df["24h %"] = np.nan
if "7d %"  not in df.columns: df["7d %"]  = np.nan
if pct24h: df["24h %"] = df["Symbol"].map(pct24h)
if pct7d:  df["7d %"]  = df["Symbol"].map(pct7d)

# ================= AHORA SÍ: SPLIT POR WALLET (DESPUÉS DE MÉTRICAS) =================
cold_aliases = {"TREZOR","HODL","VAULT","COLD"}
df["Wallet_norm"] = df["Wallet"].astype(str).str.upper().str.strip()
df_cold = df[df["Wallet_norm"].isin(cold_aliases)].copy()
df_hot  = df[~df["Wallet_norm"].isin(cold_aliases)].copy()

# ================= CONSOLIDADO =================
def pick_nearest(series, price):
    try:
        vals = [float(x) for x in series if pd.notna(x) and float(x)>0]
    except Exception:
        vals = []
    if not vals or not (pd.notna(price) and price>0): return max(vals) if vals else 0.0
    d = [abs((x-price)/x) for x in vals]; return vals[int(np.argmin(d))]

def summarize_group(g: pd.DataFrame):
    cols = g.columns
    used_price = g["Used Price (USD)"] if "Used Price (USD)" in cols else pd.Series([], dtype=float)
    last_price = used_price.dropna().iloc[-1] if not used_price.dropna().empty else np.nan
    qsum  = g["Quantity"].sum() if "Quantity" in cols else 0.0
    tcost = g["Total Cost (USD)"].sum() if "Total Cost (USD)" in cols else 0.0
    cval  = g["Current Value (USD)"].sum() if "Current Value (USD)" in cols else 0.0
    avgc  = (tcost/qsum) if qsum>0 else np.nan
    tnear = pick_nearest(g["Target Price"], last_price) if "Target Price" in cols else 0.0
    snear = pick_nearest(g["Stop Price"],   last_price) if "Stop Price"   in cols else 0.0
    p24   = g["24h %"].dropna().iloc[-1] if "24h %" in cols and g["24h %"].notna().any() else np.nan
    p7    = g["7d %"].dropna().iloc[-1]  if "7d %"  in cols and g["7d %"].notna().any()  else np.nan
    return pd.Series({
        "Quantity": qsum, "Avg Cost (USD)": avgc, "Used Price (USD)": last_price,
        "Current Value (USD)": cval, "Total Cost (USD)": tcost,
        "Target (nearest)": tnear, "Stop (nearest)": snear, "24h %": p24, "7d %": p7
    })

def make_cons(dfin: pd.DataFrame) -> pd.DataFrame:
    if dfin.empty:
        return pd.DataFrame(columns=["Symbol","Quantity","Avg Cost (USD)","Used Price (USD)","Current Value (USD)","Total Cost (USD)","Target (nearest)","Stop (nearest)","24h %","7d %"])
    cols_for_group = ["Quantity","Total Cost (USD)","Avg Cost (USD)","Used Price (USD)","Current Value (USD)","Target Price","Stop Price","24h %","7d %"]
    existing_cols = [c for c in cols_for_group if c in dfin.columns]
    gb = dfin.groupby("Symbol", group_keys=False)[existing_cols]
    try:    cons_ = gb.apply(summarize_group, include_groups=False).reset_index()
    except TypeError:
        cons_ = gb.apply(summarize_group).reset_index()
    cons_["P&L (USD)"] = cons_["Current Value (USD)"] - cons_["Total Cost (USD)"]
    cons_["P&L %"] = np.where(cons_["Total Cost (USD)"]>0, 100*cons_["P&L (USD)"]/cons_["Total Cost (USD)"], 0.0)
    total_val_local = float(cons_["Current Value (USD)"].sum(skipna=True))
    cons_["% Portfolio"] = np.where(total_val_local>0, 100*cons_["Current Value (USD)"]/total_val_local, 0.0)
    return cons_

# vistas (global y operativa)
cons_total = make_cons(df)       # frío + caliente
cons_hot   = make_cons(df_hot)   # SOLO Trading
cons       = cons_hot.copy()     # todo lo operativo usa HOT

# ================= BADGES =================
def chip_status(x):
    if x < -2: return '<span class="chip lose">▼ Pérdida</span>'
    if x >  2: return '<span class="chip win">▲ Ganancia</span>'
    return '<span class="chip flat">≈ Neutro</span>'

def chip_signals(row, band):
    p, t, s_ = row["Used Price (USD)"], row["Target (nearest)"], row["Stop (nearest)"]
    chips = []
    if pd.notna(p) and t>0 and abs((t-p)/t*100) <= band: chips.append('<span class="chip tgt">🎯</span>')
    if pd.notna(p) and s_>0 and abs((p-s_)/s_*100) <= band: chips.append('<span class="chip stop">🛑</span>')
    return " ".join(chips)

@st.cache_data(ttl=60*10)
def trending_cached():
    try: return coingecko_trending_symbols()
    except: return set()

trend_set = trending_cached() or set()
cons["Trending"] = cons["Symbol"].apply(lambda s: '<span class="chip trend">📈</span>' if s in trend_set else '')
cons["Unusual"]  = cons["24h %"].apply(lambda v: '<span class="chip unusual">⚡</span>' if (pd.notna(v) and abs(v)>=unusual_thr) else '')
cons["Status"]   = cons["P&L %"].apply(chip_status)
cons["Signals"]  = cons.apply(lambda r: chip_signals(r, band), axis=1)
cons["Invalid levels"] = (cons["Stop (nearest)"] >= cons["Target (nearest)"]) & (cons["Stop (nearest)"]>0) & (cons["Target (nearest)"]>0)
cons["StatusErr"] = np.where(cons["Invalid levels"], '<span class="chip err">config error</span>', '')

# ================= KPIs =================
k1,k2,k3,k4 = st.columns(4)
total_val_global  = float(cons_total["Current Value (USD)"].sum(skipna=True))
unreal_global     = float(cons_total["P&L (USD)"].sum(skipna=True))
with k1: st.markdown(f'<div class="kpi">**Total Value**<br/><span style="font-size:1.4rem">$ {total_val_global:,.2f}</span></div>', unsafe_allow_html=True)
with k2: st.markdown(f'<div class="kpi">**Unrealized P&L**<br/><span style="font-size:1.4rem">$ {unreal_global:,.2f}</span></div>', unsafe_allow_html=True)
with k3: st.markdown(f'<div class="kpi">**Positions (HOT)**<br/><span style="font-size:1.4rem">{len(cons)}</span></div>', unsafe_allow_html=True)
fg = None
try: fg = altme_fng()
except: pass
with k4:
    if fg: st.markdown(f'<div class="kpi">**Fear & Greed**<br/><span style="font-size:1.2rem">{fg["value"]} · {fg["classification"]}</span></div>', unsafe_allow_html=True)
    else:  st.markdown('<div class="kpi">**Fear & Greed**<br/><span style="font-size:1.2rem">n/a</span></div>', unsafe_allow_html=True)

# ================= WALLET DONUT (global) =================
st.subheader("Wallet breakdown (global)")
if df["Wallet"].astype(str).str.strip().any():
    wallets = df.copy(); wallets["wallet"] = wallets["Wallet"].astype(str).replace({"": "(Unassigned)"})
    agg = wallets.groupby("wallet", as_index=False)["Current Value (USD)"].sum().sort_values("Current Value (USD)", ascending=False)
    if len(agg) and agg["Current Value (USD)"].sum() > 0:
        try:
            import plotly.express as px
            fig = px.pie(agg, values="Current Value (USD)", names="wallet", hole=0.55)
            fig.update_layout(height=340, margin=dict(l=0,r=0,t=0,b=0), legend=dict(orientation="h", y=-0.08))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.caption(f"No pude renderizar el gráfico (plotly no instalado): {e}")
else:
    st.caption("Tu CSV no tiene 'Wallet' o viene vacío.")

# ================= CONSOLIDATED (HOT) =================
st.subheader("Consolidated — Trading (HOT only)")
cons_view = cons.copy()
cons_view["Status / Signals"] = cons_view["Status"] + " " + cons_view["Signals"] + " " + cons_view["StatusErr"] + " " + cons_view["Trending"] + " " + cons_view["Unusual"]
show = cons_view[["Symbol","Quantity","Avg Cost (USD)","Used Price (USD)","Current Value (USD)","P&L (USD)","P&L %","% Portfolio","Status / Signals"]].sort_values("Current Value (USD)", ascending=False)
st.markdown(show.style.format({
    "Quantity":"{:.8f}","Avg Cost (USD)":"${:,.2f}","Used Price (USD)":"${:,.2f}",
    "Current Value (USD)":"${:,.2f}","P&L (USD)":"${:,.2f}","P&L %":"{:.2f}%","% Portfolio":"{:.2f}%"
}).hide(axis='index').to_html(escape=False), unsafe_allow_html=True)

# ================= ALERTS =================
st.subheader("Alerts — crossed targets/stops")
alert_rows = []
for _, r in cons.iterrows():
    p, t, s_ = r["Used Price (USD)"], r["Target (nearest)"], r["Stop (nearest)"]
    if pd.notna(p) and t>0 and p >= t:  alert_rows.append({"Symbol":r["Symbol"],"Type":"🎯 Target HIT","Price":p,"Level":t,"Δ%":(p-t)/t*100})
    if pd.notna(p) and s_>0 and p <= s_: alert_rows.append({"Symbol":r["Symbol"],"Type":"🛑 Stop HIT","Price":p,"Level":s_,"Δ%":(p-s_)/s_*100})
crossed_df = pd.DataFrame(alert_rows)
if not crossed_df.empty:
    st.dataframe(crossed_df.sort_values("Symbol").style.format({"Price":"${:,.2f}","Level":"${:,.2f}","Δ%":"{:+.2f}%"}).hide(axis='index'),
                 use_container_width=True, height=220)
else:
    st.caption("Sin cruces de target/stop.")

# ================= NEAR & MOVERS =================
b1, b2 = st.columns(2)
with b1:
    st.subheader("Near Targets/Stops")
    near = cons[(cons["Used Price (USD)"].notna())].copy()
    near["near_target"] = np.where((near["Target (nearest)"]>0) & (near["Used Price (USD)"]>0) & (np.abs((near["Target (nearest)"]-near["Used Price (USD)"])/near["Target (nearest)"]*100) <= band), True, False)
    near["near_stop"]   = np.where((near["Stop (nearest)"]>0) & (near["Used Price (USD)"]>0) & (np.abs((near["Used Price (USD)"]-near["Stop (nearest)"])/near["Stop (nearest)"]*100) <= band), True, False)
    crossed_syms = set(crossed_df["Symbol"].unique()) if not crossed_df.empty else set()
    near = near[~near["Symbol"].isin(crossed_syms)]
    near = near[(near["near_target"]) | (near["near_stop"])]
    if not near.empty:
        near["Alert"] = np.where(near["near_target"] & near["near_stop"], "🎯🛑", np.where(near["near_target"], "🎯", "🛑"))
        near["Δ to Target %"] = np.where(near["Target (nearest)"]>0, (near["Target (nearest)"]-near["Used Price (USD)"])/near["Target (nearest)"]*100, np.nan)
        near["Δ to Stop %"]   = np.where(near["Stop (nearest)"]>0, (near["Used Price (USD)"]-near["Stop (nearest)"])/near["Stop (nearest)"]*100, np.nan)
        table = near[["Alert","Symbol","Used Price (USD)","Target (nearest)","Δ to Target %","Stop (nearest)","Δ to Stop %"]].sort_values("Symbol")
        st.dataframe(table.style.format({"Used Price (USD)":"${:,.2f}","Target (nearest)":"${:,.2f}","Δ to Target %":"{:+.2f}%","Stop (nearest)":"${:,.2f}","Δ to Stop %":"{:+.2f}%"}).hide(axis='index'),
                     use_container_width=True, height=280)
    else:
        st.caption("Sin activos cerca de targets/stops.")
with b2:
    st.subheader("Top Movers 24h (umbral + trending + unusual)")
    movers = cons.dropna(subset=["24h %"]).copy()
    movers["abs24"] = movers["24h %"].abs()
    movers = movers[movers["abs24"] >= mover_thr]
    movers = movers.sort_values("24h %", ascending=False)[["Symbol","Used Price (USD)","24h %","% Portfolio"]].head(12)
    if not movers.empty:
        mv = movers.copy()
        mv["Trending"] = mv["Symbol"].apply(lambda s: "📈" if s in trend_set else "")
        mv["Unusual"]  = mv["24h %"].apply(lambda v: "⚡" if (pd.notna(v) and abs(v)>=unusual_thr) else "")
        st.dataframe(mv[["Trending","Unusual","Symbol","Used Price (USD)","24h %","% Portfolio"]]
                     .style.format({"Used Price (USD)":"${:,.2f}","24h %":"{:+.2f}%","% Portfolio":"{:.2f}%"}).hide(axis='index'),
                     use_container_width=True, height=360)
    else:
        st.caption("Sin movers que superen el umbral.")

# ================= FREE NEWS =================
@st.cache_data(ttl=60*15, show_spinner=False)
def free_headlines(limit=10):
    feeds = ["https://www.coindesk.com/arc/outboundfeeds/rss/","https://cointelegraph.com/rss"]
    titles = []
    for url in feeds:
        try:
            r = requests.get(url, timeout=20); r.raise_for_status()
            for line in r.text.splitlines():
                if "<title>" in line and "</title>" in line:
                    t = line.split("<title>")[1].split("</title>")[0].strip()
                    if t and t.lower() not in ["coindesk","cointelegraph"]: titles.append(t)
        except Exception: pass
    return titles[:limit]

# ================= OPENAI ADVISOR =================
st.markdown("---"); st.subheader("🤖 OpenAI Advisor (plan + sizing)")
use_openai   = st.checkbox("Usar OpenAI para generar plan/sugerencias", value=bool(st.session_state.get("openai_key", OPENAI_KEY)))
risk_per_trd = st.number_input("Riesgo máximo por trade (%)", 0.1, 10.0, 2.0, 0.1)
rebalance_pct= st.number_input("Rebalance objetivo (%)", 0.0, 50.0, 10.0, 0.5)
dca_weekly   = st.number_input("DCA semanal (USD)", 0.0, 5000.0, 100.0, 10.0)
manual_news  = st.text_area("Titulares manuales (uno por línea)", height=100, placeholder="ETF inflows récord en BTC...\nETH prepara hard fork...")
add_free_news= st.checkbox("Agregar titulares gratis (Coindesk/Cointelegraph)", value=True)

def build_context():
    compact = cons_view[["Symbol","Quantity","Used Price (USD)","P&L %","% Portfolio"]].copy()
    compact = compact.fillna("").to_dict(orient="records")
    fg = None
    try: fg = altme_fng()
    except: pass
    news = []
    if manual_news.strip(): news.extend([s.strip() for s in manual_news.splitlines() if s.strip()])
    if add_free_news:
        try: news.extend(free_headlines(8))
        except: pass
    return {
        "portfolio": compact,
        "fear_greed": fg,
        "trending": list(trend_set)[:10],
        "news": news,
        "params": {
            "riesgo_por_trade_pct": risk_per_trd,
            "rebalance_objetivo_pct": rebalance_pct,
            "dca_semanal_usd": dca_weekly,
            "umbral_movers_pct": mover_thr,
            "umbral_unusual_pct": unusual_thr
        }
    }

def openai_advice(context):
    key = st.session_state.get("openai_key", OPENAI_KEY)
    if not key: return "Falta OPENAI_API_KEY."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        sys = "Eres un asesor de portafolios cripto. Responde en español, claro y accionable, con bullets. Incluye racional y riesgos."
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[
                {"role":"system","content":sys},
                {"role":"user","content":textwrap.dedent(f"""
                Con el siguiente contexto, genera:
                1) 3–6 oportunidades/alertas priorizadas (compra, take-profit, riesgo),
                2) tamaños ($) usando el riesgo_por_trade_pct,
                3) qué NO hacer esta semana,
                4) checklist diario (5–7 bullets con niveles).
                CONTEXTO:
                {context}
                """)}
            ]
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error llamando a OpenAI: {e}"

colL, colR = st.columns([1,1])
with colL:
    st.markdown("**Contexto para OpenAI:**")
    ctx = build_context()
    st.json(ctx, expanded=False)
with colR:
    if use_openai:
        k = (st.session_state.get("openai_key") or OPENAI_KEY)
        if not k: st.error("Falta OPENAI_API_KEY.")
        elif st.button("⚡ Generar plan con OpenAI"):
            st.markdown(openai_advice(ctx))
    else:
        st.caption("Activa 'Usar OpenAI...' para generar un plan con tu API key.")

# === INDICATORS PANEL (DROP-IN, PEGAR AL FINAL DEL DASHBOARD) ================
import requests

@st.cache_data(ttl=60*10, show_spinner=False)
def _indicators_multi(symbols: list[str], api_host: str, api_key: str, limit: int = 250) -> dict:
    if not symbols: return {}
    url = f"{api_host.rstrip('/')}/multi_indicators"
    headers = {"x-api-key": api_key}
    q = ",".join(symbols)
    r = requests.get(url, headers=headers, params={"symbols": q, "limit": limit}, timeout=30)
    r.raise_for_status()
    js = r.json()
    return js.get("data", {}) if isinstance(js, dict) else {}

def _render_indicators_panel():
    api_host = st.secrets.get("API_HOST", os.getenv("API_HOST", "https://appestrategia.onrender.com"))
    api_key  = st.secrets.get("API_KEY",  os.getenv("API_KEY", ""))
    st.markdown("---")
    st.subheader("📊 Indicators (HOT • Micro-API)")
    if not api_key:
        st.info("Configura API_KEY (Secrets o Environment) para consultar tu Micro-API.")
        return
    try:
        syms = [s for s in cons["Symbol"].dropna().unique().tolist() if s]
    except Exception:
        st.warning("No HOT table found (variable 'cons').")
        return

    data = _indicators_multi(syms, api_host, api_key)
    if not data:
        st.caption("Sin indicadores (Micro-API no respondió o sin datos).")
        return

    rows = []
    for s in syms:
        d = data.get(s) or {}
        rows.append({
            "Symbol": s,
            "Close": d.get("close"),
            "SMA50": d.get("sma50"),
            "SMA200": d.get("sma200"),
            "ATR20": d.get("atr20"),
            "↑SMA50": d.get("above_sma50"),
            "↑SMA200": d.get("above_sma200"),
        })
    tbl = pd.DataFrame(rows)
    st.dataframe(
        tbl.style.format({"Close":"${:,.2f}","SMA50":"${:,.2f}","SMA200":"${:,.2f}","ATR20":"${:,.2f}"}).hide(axis='index'),
        use_container_width=True, height=260
    )

# Auto-render del panel al final del script
try:
    _render_indicators_panel()
except Exception as _e:
    st.caption(f"Indicators panel: {_e}")
# ============================================================================#
# ==== REBUILD HOT CONSOLIDATION (DROP-IN REPLACEMENT) =======================
import re, numpy as np, pandas as pd

COLD_NAMES = {"TREZOR", "HODL", "VAULT", "COLD", "TREZOR/VAULT"}

def _money(x):
    s = str(x).strip()
    if s in ("", "nan", "None", "—", "-", "null"): 
        return np.nan
    try:
        # quita $, comas, espacios; respeta signo y punto decimal
        s = re.sub(r"[^\d\.\-]", "", s)
        return float(s)
    except Exception:
        return np.nan

def rebuild_hot_consolidation(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Consolida SOLO HOT con limpieza robusta y fallback de Used Price:
      Used Price = Price(USD) si existe → Avg Cost(USD) → 0.0
    Calcula Current Value, P&L, P&L%, %Portfolio.
    """
    df = df_raw.copy()

    # columnas mínimas
    needed = ["Symbol","Quantity","Total Cost (USD)","Avg Cost (USD)","Price (USD)","Wallet"]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    # limpieza numérica
    for c in ["Quantity","Total Cost (USD)","Avg Cost (USD)","Price (USD)"]:
        df[c] = df[c].apply(_money)

    # HOT = todo lo que NO esté marcado como COLD
    wallet_upper = df["Wallet"].fillna("").astype(str).str.upper()
    hot = df[~wallet_upper.isin(COLD_NAMES)].copy()

    # Used Price robusto (NO usa Total Cost)
    used_price = hot["Price (USD)"]
    used_price = used_price.where(used_price.notna(), hot["Avg Cost (USD)"])
    used_price = used_price.fillna(0.0)

    hot["Used Price (USD)"] = used_price
    hot["Current Value (USD)"] = (hot["Quantity"].fillna(0.0) * hot["Used Price (USD)"]).fillna(0.0)

    # groupby por símbolo
    g = (
        hot.groupby("Symbol", dropna=True, as_index=False)
           .agg({
               "Quantity": "sum",
               "Total Cost (USD)": "sum",
               "Used Price (USD)": "last",     # último precio usado
               "Current Value (USD)": "sum"
           })
    )

    # avg cost, P&L, P&L%, %Portfolio
    g["Avg Cost (USD)"] = np.where(g["Quantity"]>0, g["Total Cost (USD)"]/g["Quantity"], np.nan)
    g["P&L (USD)"] = g["Current Value (USD)"] - g["Total Cost (USD)"]
    g["P&L %"] = np.where(g["Total Cost (USD)"]>0, 100*g["P&L (USD)"]/g["Total Cost (USD)"], 0.0)
    tot = float(g["Current Value (USD)"].sum())
    g["% Portfolio"] = np.where(tot>0, 100*g["Current Value (USD)"]/tot, 0.0)

    # sanity check: posibles “usos de precio” sospechosos
    suspicious = g[(g["Avg Cost (USD)"].notna()) & (g["Used Price (USD)"] > 50 * g["Avg Cost (USD)"])]
    if len(suspicious) > 0:
        import streamlit as st
        with st.expander("⚠️ Sanity check (posibles Used Price anómalos)"):
            st.dataframe(suspicious, use_container_width=True)

    return g
# ===========================================================================#
