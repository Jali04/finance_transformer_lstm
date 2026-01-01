# src/evaluate.py
import json, pickle, math
import numpy as np
import pandas as pd
import torch
import yfinance as yf
import matplotlib.pyplot as plt
from .features import build_feature_frame
from .model import TimeSeriesTransformer
from .metrics import mae, rmse, directional_accuracy, matthews_corr

@torch.no_grad()
def evaluate_2024_and_forecast_2025():
    with open("artifacts/config.json") as f: cfg = json.load(f)["cfg"]
    with open("artifacts/scaler.pkl", "rb") as f: sc_data = pickle.load(f)
    scaler, feat_cols = sc_data["scaler"], sc_data["feat_cols"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw = yf.download(cfg["ticker"], start=cfg["start"], end=cfg["end"], auto_adjust=True, progress=False)
    df = build_feature_frame(raw)
    
    # Target für Metriken
    close = df["Close"]
    df["y"] = np.log(close.shift(-1) / close)
    df = df.dropna()

    X_full = scaler.transform(df[feat_cols]).astype("float32")
    y_full = df["y"].values
    idx = df.index

    model = TimeSeriesTransformer(in_features=len(feat_cols), d_model=cfg["d_model"], 
                                  nhead=cfg["nhead"], num_layers=cfg["num_layers"], 
                                  dim_ff=cfg["dim_ff"], dropout=cfg["dropout"]).to(device)
    model.load_state_dict(torch.load("artifacts/model.pt", map_location=device))
    model.eval()

    # Evaluation 2024
    test_mask = (idx >= "2024-01-01") & (idx <= "2024-12-31")
    preds, gts, dates = [], [], []
    
    for i in range(len(df)):
        if not test_mask[i] or i < cfg["seq_len"]: continue
        window = X_full[i - cfg["seq_len"]: i]
        x = torch.tensor(window).unsqueeze(0).to(device)
        preds.append(model(x).item())
        gts.append(y_full[i])
        dates.append(idx[i])

    # Metriken & Plots
    res = pd.DataFrame({"date": dates, "y_true": gts, "y_pred": preds})
    print("2024 Directional Accuracy:", directional_accuracy(gts, preds))
    
    # Strat: Long if pred > 0, else Short
    res["strat_ret"] = np.sign(res["y_pred"]) * res["y_true"]
    res["equity"] = np.exp(res["strat_ret"].cumsum())
    
    plt.figure(figsize=(10, 5))
    plt.plot(res["date"], res["equity"])
    plt.title(f"Equity Curve 2024 - {cfg['ticker']}")
    plt.savefig("artifacts/equity_2024.png")
    
    # Forecast 2025
    last_window = X_full[-cfg["seq_len"]:]
    f_preds = []
    curr_window = last_window.copy()
    for _ in range(20):
        x = torch.tensor(curr_window).unsqueeze(0).to(device)
        p = model(x).item()
        f_preds.append(p)
        # Für einen echten autoregressiven Forecast müssten hier Features 
        # für t+1 berechnet werden. Hier vereinfacht:
        curr_window = np.roll(curr_window, -1, axis=0)
        curr_window[-1, 0] = p # Setze pred return als neues ret1 feature

    forecast = pd.DataFrame({"day": range(1, 21), "pred_return": f_preds})
    forecast.to_csv("artifacts/forecast_2025.csv", index=False)
    print("Forecast 2025 gespeichert.")

if __name__ == "__main__":
    evaluate_2024_and_forecast_2025()