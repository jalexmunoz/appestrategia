# app.py
import os, math, time, statistics
from datetime import datetime
from typing import List, Dict
import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

API_KEY = os.getenv("API_KEY", "cambia-esto")
BINANCE = "https://api.binance.com/api/v3/klines"

app = FastAPI(title="Crypto Signals Micro API", version="1.0")

def verify_key(x_api_key: str | None):
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Bad API key")

def atr20(candles: List[Dict]) -> float:
    # candles: list of dict with 'h','l','c'
    trs = []
    for i in range(1, len(candles)):
        h,l,c_prev = candles[i]["h"], candles[i]["l"], candles[i-1]["c"]
        tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
        trs.append(tr)
    return sum(trs[-20:]) / min(20, len(trs)) if trs else 0.0

def sma(vals: List[float], n: int) -> float:
    return sum(vals[-n:]) / min(n, len(vals)) if vals else 0.0

def to_ohlc(raw):
    out=[]
    for k in raw:
        out.append({
            "t": datetime.utcfromtimestamp(k[0]/1000).strftime("%Y-%m-%d"),
            "o": float(k[1]), "h": float(k[2]), "l": float(k[3]), "c": float(k[4])
        })
    return out

async def fetch_klines(symbol: str, limit: int=250):
    params = {"symbol": symbol.upper(), "interval":"1d", "limit":limit}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(BINANCE, params=params)
        r.raise_for_status()
        return to_ohlc(r.json())

@app.get("/ohlc")
async def get_ohlc(symbol: str, limit: int=250, x_api_key: str | None = Header(default=None)):
    verify_key(x_api_key)
    data = await fetch_klines(symbol, limit)
    return JSONResponse(data)

@app.get("/multi_indicators")
async def multi_indicators(symbols: str, limit: int=250, x_api_key: str | None = Header(default=None)):
    verify_key(x_api_key)
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    out={}
    for s in syms:
        ohlc = await fetch_klines(s, limit)
        closes = [c["c"] for c in ohlc]
        res = {
            "close": closes[-1],
            "sma50":  sma(closes, 50),
            "sma200": sma(closes, 200),
            "atr20":  atr20(ohlc),
        }
        res["above_sma50"]  = res["close"] > res["sma50"]  > 0
        res["above_sma200"] = res["close"] > res["sma200"] > 0
        out[s] = res
    return JSONResponse(out)
