# src/features.py
import pandas as pd
import numpy as np

def _rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
    down = -delta.clip(upper=0).ewm(alpha=1/window, adjust=False).mean()
    return 100 - (100 / (1 + up / (down + 1e-12)))

def build_feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    
    out = pd.DataFrame(index=df.index)
    c = df["Close"].astype("float64")
    h = df["High"].astype("float64")
    l = df["Low"].astype("float64")
    v = df["Volume"].astype("float64")

    # Stationäre Features
    out["ret1"] = np.log(c / c.shift(1))
    out["ret5"] = np.log(c / c.shift(5))
    
    # Relative Preisindikatoren (Wichtig für NN Stabilität)
    for w in [20, 50]:
        out[f"sma{w}_rel"] = (c.rolling(w).mean() / c) - 1.0
        out[f"std{w}"] = out["ret1"].rolling(w).std()

    # Momentum & Trend
    out["rsi14"] = _rsi(c, 14)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    out["macd_norm"] = (ema12 - ema26) / c
    
    # Volatilität & Volumen
    out["hl_range"] = (h - l) / c
    out["vol_z20"] = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-12)

    # Kalender (optional aber oft hilfreich)
    out["dow"] = out.index.dayofweek
    out = pd.get_dummies(out, columns=["dow"], drop_first=True)

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out["Close"] = c # Für das Target in train.py
    return out.sort_index()