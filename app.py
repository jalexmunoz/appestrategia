# app.py — Micro-API para tu GPT (CMC principal + Binance backup)
import os
import logging
from datetime import datetime
from typing import List, Dict

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

# -------------------- Config & Logs --------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("crypto-microapi")

API_KEY = os.getenv("API_KEY", "alex-crypto-123")             # clave para tu Action (header x-api-key)
CMC_API_KEY = os.getenv("CMC_API_KEY", "0205e7f3-a24f-4749-8447-8e644103401d")                # tu clave de CoinMarketCap

# -------------------- Proveedor principal: CoinMarketCap --------------------
CMC_BASE = "https://pro-api.coinmarketcap.com"
CMC_HEADERS = {
    "X-CMC_PRO_API_KEY": CMC_API_KEY,
    "Accept": "application/json",
    "User-Agent": "crypto-microapi/1.0"
}

# Cache simple en memoria para mapear símbolo -> id de CMC (evita pedirlo siempre)
CMC_ID_CACHE: Dict[str, int] = {}

def _sma(vals: List[float], n: int) -> float:
    if not vals:
        return 0.0
    m = min(n, len(vals))
    return sum(vals[-m:]) / m

def _atr20(candles: List[Dict]) -> float:
    trs = []
    for i in range(1, len(candles)):
        h = candles[i]["h"]
        l = candles[i]["l"]
        c_prev = candles[i - 1]["c"]
        tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
        trs.append(tr)
    if not trs:
        return 0.0
    m = min(20, len(trs))
    return sum(trs[-m:]) / m

def _normalize_for_cmc(symbol: str) -> str:
    """CMC usa tickers tipo BTC, ETH, SOL… No hace falta sufijo USDT/USD."""
    s = symbol.upper().strip()
    if s.endswith("USDT"):
        return s[:-4]
    if s.endswith("USD"):
        return s[:-3]
    return s

async def _cmc_get_id(symbol: str) -> int | None:
    """Busca el ID de CMC para un símbolo (elige el primero por market cap)."""
    sym = _normalize_for_cmc(symbol)
    if sym in CMC_ID_CACHE:
        return CMC_ID_CACHE[sym]
    if not CMC_API_KEY:
        return None
    url = f"{CMC_BASE}/v1/cryptocurrency/map"
    params = {"symbol": sym}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, params=params, headers=CMC_HEADERS)
        if r.status_code >= 400:
            logger.warning("CMC map fallo %s: %s %s", sym, r.status_code, r.text[:200])
            return None
        data = r.json().get("data") or []
        if not data:
            return None
        # Tomamos el primero (suele ser el de mayor relevancia)
        id_ = int(data[0]["id"])
        CMC_ID_CACHE[sym] = id_
        return id_

async def cmc_ohlc(symbol: str, limit: int = 250) -> List[Dict]:
    """OHLC diario desde CMC v2 (intenta por símbolo y luego por id)."""
    if not CMC_API_KEY:
        raise HTTPException(status_code=500, detail="CMC_API_KEY no configurada")

    async def _request(params):
        url = f"{CMC_BASE}/v2/cryptocurrency/ohlcv/historical"
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(url, params=params, headers=CMC_HEADERS)
            if r.status_code >= 400:
                raise HTTPException(status_code=502, detail=f"CMC error {r.status_code}: {r.text[:400]}")
            js = r.json()
            quotes = js.get("data", {}).get("quotes") or []
            out = []
            for q in quotes[-limit:]:
                usd = q["quote"]["USD"]
                out.append({
                    "t": (q.get("time_open") or q.get("timestamp") or "")[:10],  # YYYY-MM-DD
                    "o": float(usd["open"]),
                    "h": float(usd["high"]),
                    "l": float(usd["low"]),
                    "c": float(usd["close"]),
                })
            if not out:
                raise HTTPException(status_code=502, detail="CMC sin datos")
            return out

    # 1) Por símbolo (BTC, ETH, SOL…)
    sym = _normalize_for_cmc(symbol)
    try:
        return await _request({
            "symbol": sym, "convert": "USD",
            "time_period": "daily", "interval": "daily", "count": limit
        })
    except HTTPException as he:
        logger.warning("CMC por símbolo fallo (%s): %s", sym, he.detail)

    # 2) Por ID
    id_ = await _cmc_get_id(sym)
    if id_ is not None:
        logger.info("Reintentando CMC por id: %s -> %s", sym, id_)
        return await _request({
            "id": id_, "convert": "USD",
            "time_period": "daily", "interval": "daily", "count": limit
        })

    # Si no hay ID tampoco, falla
    raise HTTPException(status_code=502, detail=f"CMC no pudo resolver símbolo {sym}")

# -------------------- Respaldo: Binance (mirror) --------------------
BINANCE_HOSTS = [
    "https://data-api.binance.vision",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api.binance.com",
    "https://api.binance.us",
]
KLINES_PATH = "/api/v3/klines"
BINANCE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; FastAPI/1.0; +https://onrender.com)",
    "Accept": "application/json",
}

def _candidate_symbols(symbol: str, host: str):
    s = symbol.upper().strip()
    yield s
    if "binance.us" in host and s.endswith("USDT"):
        yield s[:-4] + "USD"  # BTCUSDT -> BTCUSD (en binance.us muchos pares son USD)

def _to_ohlc_binance(raw):
    out=[]
    for k in raw:
        out.append({
            "t": datetime.utcfromtimestamp(k[0] / 1000).strftime("%Y-%m-%d"),
            "o": float(k[1]), "h": float(k[2]), "l": float(k[3]), "c": float(k[4])
        })
    return out

async def fetch_klines_binance_backup(symbol: str, limit: int = 250) -> List[Dict]:
    params_base = {"interval": "1d", "limit": limit}
    last_err = None
    for host in BINANCE_HOSTS:
        url = host + KLINES_PATH
        for sym in _candidate_symbols(symbol, host):
            params = {"symbol": sym, **params_base}
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
                    r = await client.get(url, params=params, headers=BINANCE_HEADERS)
                    if r.status_code >= 400:
                        txt = r.text[:400]
                        logger.warning("Binance %s %s → %s %s", host, sym, r.status_code, txt)
                        last_err = (r.status_code, f"{host} {sym}: {txt}")
                        continue
                    data = r.json()
                    return _to_ohlc_binance(data)
            except Exception as e:
                logger.exception("Error llamando %s para %s", host, sym)
                last_err = (599, f"{host} {sym}: {str(e)}")
                continue
    status, detail = last_err or (502, "Unknown upstream error")
    raise HTTPException(status_code=502, detail=f"Binance upstream error for {symbol}: {status} {detail}")

# -------------------- Selección de proveedor --------------------
async def fetch_klines(symbol: str, limit: int = 250) -> List[Dict]:
    """Intenta CMC primero; si falla, usa Binance backup."""
    try:
        return await cmc_ohlc(symbol, limit)
    except HTTPException as he:
        logger.warning("CMC fallo para %s: %s", symbol, he.detail)
    except Exception as e:
        logger.exception("CMC exception para %s: %s", symbol, e)

    # Respaldo
    return await fetch_klines_binance_backup(symbol, limit)

# -------------------- FastAPI endpoints --------------------
app = FastAPI(title="Crypto Signals Micro API", version="1.1 (CMC+Backup)")

def _verify_key(x_api_key: str | None):
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Bad API key")

@app.get("/")
async def root():
    return {"ok": True, "service": "crypto-signals", "version": "1.1"}

@app.get("/health")
async def health():
    return {"ok": True, "cmc": bool(CMC_API_KEY)}

@app.get("/ohlc")
async def get_ohlc(symbol: str, limit: int = 250, x_api_key: str | None = Header(default=None)):
    _verify_key(x_api_key)
    try:
        data = await fetch_klines(symbol, limit)
        return JSONResponse(data)
    except HTTPException as he:
        return JSONResponse({"error": he.detail}, status_code=he.status_code)
    except Exception as e:
        logger.exception("/ohlc error")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/multi_indicators")
async def multi_indicators(symbols: str, limit: int = 250, x_api_key: str | None = Header(default=None)):
    _verify_key(x_api_key)
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    out, errors = {}, {}

    for s in syms:
        try:
            ohlc = await fetch_klines(s, limit)
            closes = [c["c"] for c in ohlc]
            res = {
                "close": closes[-1],
                "sma50": _sma(closes, 50),
                "sma200": _sma(closes, 200),
                "atr20": _atr20(ohlc),
            }
            res["above_sma50"]  = res["close"] > res["sma50"]  > 0
            res["above_sma200"] = res["close"] > res["sma200"] > 0
            out[s] = res
        except HTTPException as he:
            logger.warning("Indicador fallo %s: %s", s, he.detail)
            errors[s] = he.detail
        except Exception as e:
            logger.exception("Indicador exception %s", s)
            errors[s] = str(e)

    return JSONResponse({"data": out, "errors": errors})
