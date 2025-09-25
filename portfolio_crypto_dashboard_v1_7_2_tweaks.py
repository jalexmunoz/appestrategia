
import os, json, time, requests, math
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

st.set_page_config(page_title="Crypto Dashboard — v1.7.2", layout="wide")

# ---- Dark theme + compact sidebar CSS ----
st.markdown("""
<style>
:root, .stApp {
  --bg: #0B1220; --panel: #0F172A; --panel-2: #111827;
  --text: #e5e7eb; --muted: #94a3b8;
  --green:#10b981; --red:#ef4444; --amber:#f59e0b;
  --chip-bg: rgba(255,255,255,0.06);
}
.stApp { background: var(--bg); color: var(--text); }
[data-testid="stSidebar"] { width: 240px; min-width: 240px; }
.block-container { padding-top: 0.8rem; }
.title { font-size: 1.9rem; font-weight: 800; color: var(--text) }
.kpi { background: linear-gradient(180deg, rgba(124,58,237,0.12), rgba(124,58,237,0.05));
       border:1px solid rgba(124,58,237,0.35); padding:14px 16px; border-radius:14px; }
.subtle { color: var(--muted); }
.chip{ display:inline-block; padding:4px 10px; border-radius:999px; font-size:.78rem; font-weight:600;
       background:var(--chip-bg); border:1px solid rgba(255,255,255,.08); margin-right:6px }
.chip.win{ color:var(--green) } .chip.lose{ color:var(--red) } .chip.flat{ color:var(--amber) }
.chip.tgt{ color:#c084fc } .chip.stop{ color:#fca5a5 }
.dataframe th, .dataframe td { color: var(--text) !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🟣 Crypto Dashboard — v1.7.2</div>', unsafe_allow_html=True)
st.caption("Sidebar más angosta, **OpenAI plan de acción**, y **Near Targets/Stops** con icono de alerta y sin índice.")

# ---- Sidebar ----
api_key = os.getenv("CMC_API_KEY","")
api_key = st.sidebar.text_input("CMC API Key", value=api_key, type="password")
use_live = st.sidebar.toggle("Usar precios live de CMC", value=bool(api_key))
band = st.sidebar.slider("Cerca de Target/Stop (±%)", 1.0, 20.0, 5.0, 0.5)
cache_minutes = st.sidebar.number_input("Cache de precios (min)", 0, 120, 10)
st.sidebar.markdown("---")
use_openai = st.sidebar.toggle("Usar OPENAI para 'Qué hacer hoy'", value=bool(os.getenv("OPENAI_API_KEY","")))

# ---- File upload ----
up = st.file_uploader("Upload crypto CSV", type=["csv"])
if up is None:
    st.info("Sube tu CSV con: Symbol, Quantity, Total Cost (USD), Avg Cost (USD), Price (USD) (opcional), Target Price, Stop Price, Note, Wallet")
    st.stop()
df = pd.read_csv(up)

# ---- Normalize ----
for c in ["Quantity","Total Cost (USD)","Avg Cost (USD)","Price (USD)","Target Price","Stop Price"]:
    if c in df.columns:
        df[c] = df[c].apply(lambda x: float(str(x).replace("$","").replace(",","")) if str(x).strip() not in ["","None","nan"] else 0.0)
df["Symbol"] = df["Symbol"].astype(str).str.upper()
if "Wallet" not in df.columns: df["Wallet"] = ""

# ---- Cache quotes ----
CACHE_FN = "cmc_cache.json"
now_ts = int(time.time())
cached = {}
if cache_minutes>0 and os.path.exists(CACHE_FN):
    try: cached = json.load(open(CACHE_FN,"r"))
    except Exception: cached = {}

def get_cached(s):
    v = cached.get(s.upper())
    if not v: return None
    if now_ts - v.get("ts",0) <= cache_minutes*60:
        return v["price"], v.get("pct24h", 0.0)
    return None
def set_cached(s, price, pct):
    cached[s.upper()] = {"price":price, "pct24h":pct, "ts":now_ts}

quotes, pct24h = {}, {}
syms = df["Symbol"].dropna().unique().tolist()
need = []
for s in syms:
    hit = get_cached(s) if cache_minutes>0 else None
    if hit:
        q, p = hit; quotes[s]=q; pct24h[s]=p
    else:
        need.append(s)

if use_live and api_key and need:
    url = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest"
    headers = {"X-CMC_PRO_API_KEY": api_key}; params = {"symbol": ",".join(need), "convert": "USD"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code==200:
            data = r.json().get("data", {})
            for s, arr in data.items():
                it = arr[0] if isinstance(arr,list) and arr else arr
                price = float(it["quote"]["USD"]["price"]); ch = float(it["quote"]["USD"].get("percent_change_24h",0.0))
                quotes[s.upper()] = price; pct24h[s.upper()] = ch
                if cache_minutes>0: set_cached(s, price, ch)
        else:
            st.warning(f"CMC error: {r.status_code} - {r.text[:160]}")
    except Exception as e:
        st.warning(f"CMC request failed: {e}")

if cache_minutes>0:
    try: json.dump(cached, open(CACHE_FN,"w"))
    except Exception: pass

# ---- Metrics ----
df["Live Price (USD)"] = df["Symbol"].map(quotes) if quotes else np.nan
df["Used Price (USD)"] = df["Live Price (USD)"].fillna(df.get("Price (USD)", np.nan))
df["Current Value (USD)"] = df["Quantity"] * df["Used Price (USD)"]
total_val = float(df["Current Value (USD)"].sum(skipna=True))
total_cost = float(df["Total Cost (USD)"].sum(skipna=True))
unreal = (total_val - total_cost) if total_val and total_cost else 0.0
pnl_pct = (unreal/total_cost*100.0) if total_cost>0 else 0.0
df["P&L (USD)"] = df["Current Value (USD)"] - df["Total Cost (USD)"]
df["P&L %"] = np.where(df["Total Cost (USD)"]>0, 100*df["P&L (USD)"]/df["Total Cost (USD)"], 0.0)
df["% Portfolio"] = np.where(total_val>0, 100*df["Current Value (USD)"]/total_val, 0.0)
df["24h %"] = df["Symbol"].map(pct24h) if pct24h else np.nan

# ---- KPIs ----
k1,k2,k3 = st.columns(3)
with k1: st.markdown(f'<div class="kpi">**Total Value**<br/><span style="font-size:1.4rem">$ {total_val:,.2f}</span></div>', unsafe_allow_html=True)
with k2: st.markdown(f'<div class="kpi">**Unrealized P&L**<br/><span style="font-size:1.4rem">$ {unreal:,.2f}</span> <span class="subtle">({pnl_pct:,.2f}%)</span></div>', unsafe_allow_html=True)
with k3: st.markdown(f'<div class="kpi">**Total Cost**<br/><span style="font-size:1.4rem">$ {total_cost:,.2f}</span></div>', unsafe_allow_html=True)

# ---- Consolidation ----
def pick_nearest(series, price):
    vals = [float(x) for x in series if float(x)>0]
    if not vals or not (price and price>0): return max(vals) if vals else 0.0
    d = [abs((x-price)/x) for x in vals]; return vals[int(np.argmin(d))]

cons = (df.groupby("Symbol").apply(lambda g: pd.Series({
    "Quantity": g["Quantity"].sum(),
    "Avg Cost (USD)": (g["Total Cost (USD)"].sum()/g["Quantity"].sum()) if g["Quantity"].sum()>0 else np.nan,
    "Used Price (USD)": g["Used Price (USD)"].dropna().iloc[-1] if g["Used Price (USD)"].notna().any() else np.nan,
    "Current Value (USD)": g["Current Value (USD)"].sum(),
    "Total Cost (USD)": g["Total Cost (USD)"].sum(),
    "Target (nearest)": pick_nearest(g["Target Price"], g["Used Price (USD)"].dropna().iloc[-1] if g["Used Price (USD)"].notna().any() else 0),
    "Stop (nearest)": pick_nearest(g["Stop Price"], g["Used Price (USD)"].dropna().iloc[-1] if g["Used Price (USD)"].notna().any() else 0),
    "24h %": g["24h %"].dropna().iloc[-1] if g["24h %"].notna().any() else np.nan
}))).reset_index()

cons["P&L (USD)"] = cons["Current Value (USD)"] - cons["Total Cost (USD)"]
cons["P&L %"] = np.where(cons["Total Cost (USD)"]>0, 100*cons["P&L (USD)"]/cons["Total Cost (USD)"], 0.0)
cons["% Portfolio"] = np.where(total_val>0, 100*cons["Current Value (USD)"]/total_val, 0.0)

def chip_status(x):
    if x < -2: return '<span class="chip lose">▼ Perdida</span>'
    if x >  2: return '<span class="chip win">▲ Ganancia</span>'
    return '<span class="chip flat">≈ Neutro</span>'

def chip_signals(row, band):
    p = row["Used Price (USD)"]; tgt = row["Target (nearest)"]; stp = row["Stop (nearest)"]
    chips = []
    if tgt>0 and p>0 and abs((tgt-p)/tgt*100) <= band: chips.append('<span class="chip tgt">🎯</span>')
    if stp>0 and p>0 and abs((p-stp)/stp*100) <= band: chips.append('<span class="chip stop">🛑</span>')
    return " ".join(chips)

cons["Status"] = cons["P&L %"].apply(chip_status)
cons["Signals"] = cons.apply(lambda r: chip_signals(r, band), axis=1)

# ---- TOP: Consolidated (wide) + Wallet donut ----
left, right = st.columns([8,4])

with left:
    st.subheader("Consolidated")
    cons_view = cons.copy()
    cons_view["Status / Signals"] = cons_view["Status"] + " " + cons_view["Signals"]
    show = cons_view[["Symbol","Quantity","Avg Cost (USD)","Used Price (USD)","Current Value (USD)","P&L (USD)","P&L %","% Portfolio","Status / Signals"]].sort_values("Current Value (USD)", ascending=False)
    html = show.style.format({
        "Quantity":"{:.8f}","Avg Cost (USD)":"${:,.2f}","Used Price (USD)":"${:,.2f}",
        "Current Value (USD)":"${:,.2f}","P&L (USD)":"${:,.2f}","P&L %":"{:.2f}%","% Portfolio":"{:.2f}%"
    }).hide(axis='index').to_html(escape=False)
    st.markdown(html, unsafe_allow_html=True)

with right:
    st.subheader("Wallet breakdown")
    w = df.groupby("Wallet", as_index=False)["Current Value (USD)"].sum().rename(columns={"Current Value (USD)":"Value (USD)"})
    w = w[w["Wallet"].astype(str).str.len()>0]
    if not w.empty and total_val>0:
        w["% Portfolio"] = 100*w["Value (USD)"]/total_val
        w_sorted = w.sort_values("Value (USD)", ascending=False)
        if len(w_sorted) > 6:
            top = w_sorted.head(5).copy()
            other = pd.DataFrame({"Wallet":["Other"], "Value (USD)":[w_sorted["Value (USD)"].iloc[5:].sum()]})
            w_plot = pd.concat([top, other], ignore_index=True)
            w_plot["% Portfolio"] = 100*w_plot["Value (USD)"]/total_val
        else:
            w_plot = w_sorted.copy()
        pie = px.pie(w_plot, names="Wallet", values="Value (USD)", hole=0.45, color_discrete_sequence=px.colors.qualitative.Safe)
        pie.update_traces(textposition='inside', texttemplate="%{label}<br>%{percent:.1%}")
        pie.update_layout(paper_bgcolor="#0B1220", plot_bgcolor="#0B1220", font_color="#e5e7eb")
        st.plotly_chart(pie, use_container_width=True)
    else:
        st.caption("No hay información de Wallet en el CSV.")

# ---- BOTTOM: Near / Movers ----
b1, b2 = st.columns(2)
with b1:
    st.subheader("Near Targets/Stops")
    near = cons[(cons["Used Price (USD)"].notna())].copy()
    # flags
    near["near_target"] = np.where((near["Target (nearest)"]>0) & (near["Used Price (USD)"]>0) & (np.abs((near["Target (nearest)"]-near["Used Price (USD)"])/near["Target (nearest)"]*100) <= band), True, False)
    near["near_stop"]   = np.where((near["Stop (nearest)"]>0) & (near["Used Price (USD)"]>0) & (np.abs((near["Used Price (USD)"]-near["Stop (nearest)"])/near["Stop (nearest)"]*100) <= band), True, False)
    near = near[(near["near_target"]) | (near["near_stop"])]
    if not near.empty:
        near["Alert"] = np.where(near["near_target"] & near["near_stop"], "🎯🛑",
                          np.where(near["near_target"], "🎯", "🛑"))
        near["Δ to Target %"] = np.where(near["Target (nearest)"]>0, (near["Target (nearest)"]-near["Used Price (USD)"])/near["Target (nearest)"]*100.0, np.nan)
        near["Δ to Stop %"]   = np.where(near["Stop (nearest)"]>0, (near["Used Price (USD)"]-near["Stop (nearest)"])/near["Stop (nearest)"]*100.0, np.nan)
        table = near[["Alert","Symbol","Used Price (USD)","Target (nearest)","Δ to Target %","Stop (nearest)","Δ to Stop %"]]                .sort_values("Symbol")
        st.dataframe(
            table.style.format({
                "Used Price (USD)":"${:,.2f}","Target (nearest)":"${:,.2f}","Δ to Target %":"{:+.2f}%",
                "Stop (nearest)":"${:,.2f}","Δ to Stop %":"{:+.2f}%"
            }).hide(axis='index'),
            use_container_width=True, height=280
        )
    else:
        st.caption("Sin activos cerca de targets/stops.")
with b2:
    st.subheader("Top Movers 24h")
    movers = cons.dropna(subset=["24h %"]).sort_values("24h %", ascending=False)[["Symbol","Used Price (USD)","24h %","% Portfolio"]].head(10)
    if not movers.empty:
        st.dataframe(movers.style.format({"Used Price (USD)":"${:,.2f}","24h %":"{:+.2f}%","% Portfolio":"{:.2f}%"}).hide(axis='index'),
                     use_container_width=True, height=280)
    else:
        st.caption("Activa CMC para ver cambios 24h.")

# ---- Actionable summary (OpenAI) ----
if use_openai:
    st.subheader("Qué hacer hoy (OpenAI)")
    if st.button("🧠 Generar plan de acción"):
        try:
            from openai import OpenAI
            client = OpenAI()
            payload = {
                "total_value": float(total_val),
                "unreal_pnl": float(unreal),
                "pnl_pct": float(pnl_pct),
                "near_signals": table.to_dict(orient="records") if 'table' in locals() else [],
                "top_weights": cons.sort_values("% Portfolio", ascending=False).head(5)[["Symbol","% Portfolio","P&L %"]].to_dict(orient="records"),
            }
            prompt = [
                {"role":"system","content":"Eres un portfolio coach de cripto. Devuelve bullets claros y accionables."},
                {"role":"user","content": f"Datos del portafolio: {json.dumps(payload)}. Dame 3–5 bullets: acciones inmediatas, riesgo por wallet, rebalanceo si hay concentración y una idea DCA."}
            ]
            rsp = client.chat.completions.create(model=os.getenv("OPENAI_MODEL","gpt-4o-mini"), messages=prompt, temperature=0.3)
            text = rsp.choices[0].message.content
            st.success("Plan de acción:")
            st.markdown(text)
        except Exception as e:
            st.warning(f"No pude generar el resumen con OpenAI: {e}")
