# src/train.py
import os, json, pickle, math, random
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import matplotlib.pyplot as plt

from .features import build_feature_frame
from .dataset import WindowDataset
from .model import TimeSeriesTransformer
from .metrics import mae, rmse, directional_accuracy

CFG = dict(
    ticker="AAPL",
    start="2010-01-01",
    end="2025-01-05",
    seq_len=40,        # Etwas kürzer gegen Rauschen
    d_model=64,
    nhead=4,
    num_layers=2,
    dim_ff=128,
    dropout=0.3,
    lr=1e-4,           # Stabilere Lernrate
    weight_decay=0.05, # Starke Regularisierung
    batch_size=64,
    epochs=100,
    patience=15,
    train_end="2022-12-30",
    valid_end="2023-12-29",
    seed=42
)

def train_main(cfg=CFG):
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    os.makedirs("artifacts", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw = yf.download(cfg["ticker"], start=cfg["start"], end=cfg["end"], auto_adjust=True, progress=False)
    df = build_feature_frame(raw)
    
    # Target: log-return t -> t+1
    close = df["Close"]
    df["y"] = np.log(close.shift(-1) / close)
    df = df.dropna()

    # Split
    tr_df = df.loc[:cfg["train_end"]]
    va_df = df.loc[pd.Timestamp(cfg["train_end"]) + pd.Timedelta(days=1):cfg["valid_end"]]
    
    feat_cols = [c for c in df.columns if c not in ["y", "Close"]]
    scaler = StandardScaler().fit(tr_df[feat_cols])
    
    X_tr = scaler.transform(tr_df[feat_cols]).astype("float32")
    y_tr = tr_df["y"].values.astype("float32")
    X_va = scaler.transform(va_df[feat_cols]).astype("float32")
    y_va = va_df["y"].values.astype("float32")

    Ltr = DataLoader(WindowDataset(X_tr, y_tr, cfg["seq_len"]), batch_size=cfg["batch_size"], shuffle=True)
    Lva = DataLoader(WindowDataset(X_va, y_va, cfg["seq_len"]), batch_size=cfg["batch_size"])

    model = TimeSeriesTransformer(in_features=len(feat_cols), d_model=cfg["d_model"], 
                                  nhead=cfg["nhead"], num_layers=cfg["num_layers"], 
                                  dim_ff=cfg["dim_ff"], dropout=cfg["dropout"]).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.HuberLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_loss = float("inf")
    wait = 0
    for ep in range(cfg["epochs"]):
        model.train()
        for xb, yb in Ltr:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        va_loss, preds, gts = 0, [], []
        with torch.no_grad():
            for xb, yb in Lva:
                xb, yb = xb.to(device), yb.to(device)
                p = model(xb)
                va_loss += loss_fn(p, yb).item() * xb.size(0)
                preds.append(p.cpu().numpy()); gts.append(yb.cpu().numpy())
        
        va_loss /= len(Lva.dataset)
        scheduler.step(va_loss)
        da = directional_accuracy(np.concatenate(gts), np.concatenate(preds))
        print(f"Epoch {ep:02d} | Val Loss: {va_loss:.6f} | DirAcc: {da:.3f}")

        if va_loss < best_loss:
            best_loss = va_loss
            wait = 0
            torch.save(model.state_dict(), "artifacts/model.pt")
        else:
            wait += 1
            if wait >= cfg["patience"]: break

    with open("artifacts/config.json", "w") as f: json.dump({"cfg": cfg, "feat_cols": feat_cols}, f)
    with open("artifacts/scaler.pkl", "wb") as f: pickle.dump({"scaler": scaler, "feat_cols": feat_cols}, f)

if __name__ == "__main__":
    train_main()