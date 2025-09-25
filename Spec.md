# AppEstrategia – Micro-API + Dashboard

## Contexto
El dashboard (Streamlit) consulta a una Micro-API (FastAPI) para indicadores 1D y sugerencias OCO. Hoy hay timeouts y símbolos mal normalizados.

## Objetivo (v1.15)
- Proveer velas/indicadores estables y rápidos.
- Normalizar símbolos (BTC→BTCUSDT, etc.).
- Evitar timeouts devolviendo parciales por símbolo.
- Mantener OCO sólo de **salida spot** (SELL), Entry=N/A.

## Alcance
- **Micro-API**: proveedor de velas con preferencia Binance klines, fallback CMC OHLC; cache TTL; normalización de símbolos; timeouts cortos; endpoint `/multi_indicators` robusto.
- **Dashboard**: ya actualizado a v1.15 (solo verificaciones menores).

### Fuera de alcance
- OCO de **entrada** (bracket).
- Cambios visuales mayores en Streamlit.

## Contratos
### GET `/multi_indicators`
- `symbols`: CSV de símbolos arbitrarios (p. ej. `BTC,ETH,SOLUSDT`).
- Respuesta: `{ "data": { "BTCUSDT": { close, sma50, sma200, atr20, above_sma50, above_sma200 } }, "errors": { "XXX": "mensaje" } }`
- Reglas:
  - Normalizar cada símbolo a sufijo **USDT**.
  - Obtener velas **1d, limit=250**.
  - **Timeout total** 20s máx; por proveedor 6–8s.
  - Cache TTL **60s** por `(provider,symbol,interval,limit)`.
  - Si un símbolo falla, **no** falla todo: meter en `errors`.

### GET `/oco_suggest`
- Debe aceptar símbolos sin sufijo y normalizar a **USDT**.
- Usa close y ATR20 de velas 1d.
- Cálculo de salida: `stop = close - k*atr20`, `take = close + R*atr20`, `qty = risk_usd / (close - stop)`, `rr = (take-close)/(close-stop)`.
- Devuelve: `{ stop, take, qty, rr }`.

## Variables (.env)
- `API_KEY` (para x-api-key).
- `CMC_API_KEY` (plan actual).
- (Opcional) `MICRO_API_URL`.

## Criterios de aceptación
1. `GET /multi_indicators?symbols=BTC,ETH,SOL` responde < **2s** con `BTCUSDT/ETHUSDT/SOLUSDT` en `data` (sin timeouts).
2. `GET /oco_suggest?symbol=BTC&k=1&R=2&risk_usd=50` responde **200** con valores numéricos y `rr>0`.
3. Si un símbolo falla en CMC (403/429), aún se obtienen datos por Binance.
4. Cache TTL visible: dos llamadas consecutivas al mismo símbolo no generan segundo fetch al proveedor dentro de 60s.
5. Si todos los proveedores fallan para un símbolo, aparece en `errors` (y no tumba el endpoint).

## Archivos a tocar
- `providers.py` (nuevo): normalización, cache TTL, Binance primero, CMC fallback.
- `routes/indicators.py` (o bloque en `app.py`): endpoint `/multi_indicators` usando `providers.get_ohlc`.
- (Opcional) `/oco_suggest` para normalizar y reusar velas.
- Tests: `tests/test_indicators.py`, `tests/test_oco.py`.

## Pruebas
- Unit: normalización, cálculo SMA/ATR, rr>0.
- E2E: llamadas reales a `/multi_indicators` y `/oco_suggest` con 2–3 símbolos.

## Rollback
- Revertir PR y desactivar cache.
