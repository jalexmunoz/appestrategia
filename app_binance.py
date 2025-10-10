import os, math, logging
from datetime import datetime
from typing import List, Dict
import httpx, numpy as np
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

# ============================================================
# 🔹 CONFIGURACIÓN
# ============================================================
BINANCE_BASE = os.getenv("BINANCE_BASE", "https://api.binance.us")
API_KEY = os.getenv("API_KEY", "alex-crypto-123")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("binance-microapi")

app = FastAPI(title="Crypto Signals (Binance Only)", version="2.0")

# ============================================================
# 🔹 UTILIDADES DE CÁLCULO
# ============================================================
def _sma(values: List[float], n: int) -> float:
    if not values: return 0.0
    m = min(n, len(values))
    return sum(values[-m:]) / m

def _atr20(candles: List[Dict]) -> float:
    trs = []
    for i in range(1, len(candles)):
        h, l, c_prev = candles[i]['h'], candles[i]['l'], candles[i-1]['c']
        tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
        trs.append(tr)
    if not trs: return 0.0
    return sum(trs[-20:]) / min(20, len(trs))

# ============================================================
# 🔹 BINANCE — DESCARGA DE VELAS
# ============================================================
async def fetch_binance_ohlc(symbol: str, limit: int = 250) -> List[Dict]:
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": "1d", "limit": limit}
    headers = {"Accept": "application/json", "User-Agent": "binance-microapi/2.0"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(url, params=params, headers=headers)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Binance error {r.status_code}: {r.text[:200]}")
        data = r.json()

    out = []
    for k in data:
        out.append({
            "t": datetime.utcfromtimestamp(k[0] / 1000).strftime("%Y-%m-%d"),
            "o": float(k[1]),
            "h": float(k[2]),
            "l": float(k[3]),
            "c": float(k[4])
        })
    return out

# ============================================================
# 🔹 ENDPOINTS
# ============================================================
@app.get("/")
async def root():
    return {"ok": True, "service": "binance-signals", "version": "2.0"}

@app.get("/health")
async def health():
    return {"ok": True, "source": "binance"}

@app.get("/multi_indicators")
async def multi_indicators(symbols: str, limit: int = 250, x_api_key: str | None = Header(default=None)):
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Bad API key")

    syms = [s.strip().upper() for s in symbols.split(',') if s.strip()]
    out, errors = {}, {}

    for s in syms:
        try:
            candles = await fetch_binance_ohlc(s, limit)
            closes = [c['c'] for c in candles]
            close = closes[-1]
            sma50 = _sma(closes, 50)
            sma200 = _sma(closes, 200)
            atr20 = _atr20(candles)

            res = {
                "close": close,
                "sma50": sma50,
                "sma200": sma200,
                "atr20": atr20,
                "above_sma50": close > sma50 > 0,
                "above_sma200": close > sma200 > 0
            }
            out[s] = res
        except Exception as e:
            logger.exception("Fallo indicador %s", s)
            errors[s] = str(e)

    return JSONResponse({"data": out, "errors": errors})

# ============================================================
# 🔹 ARRANQUE LOCAL (opcional)
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))

