# Block 7 – Abschluss-Report

- **Run-Ordner:** `2026-01-01_18-20-06_lstm`
- **Erstellt (UTC):** 2026-01-01T17:21:43Z
- **Ticker/Intervall:** AAPL / 1d
- **Zeitraum:** 2010-01-01 → 2026-01-01
- **Horizon/Lookback:** H=1 / LB=60
- **Featureset:** v2 → logret_1d, logret_3d, logret_5d, realized_vol_10, bb_pos, rsi_14, macd, macd_sig, macd_diff, vol_z_20, sma_diff

## Test-Metriken
- AUROC: **0.508**, AUPRC: **0.532** (Random=0.53? Check PosRate), Brier: **0.250**
- Balanced Acc: **0.509**, MCC: **0.022**

![ROC](figures\roc_test.png)  
![PR](figures\pr_test.png)

## Backtests
- **Gross (ohne Kosten)**: Entry@t CAGR=0.164, Entry@t+1 CAGR=0.119
## Kosten-KPI (realistisch)
- **Netto (mit Kosten)**: Entry@t+1 (No-Overlap) @ 15.0 bps. CAGR=-0.128, Equity=0.746
![Equity net](figures\equity_costed.png)

## Limitations
- **Labeling:** Sensitivität gegenüber Epsilon ist hoch.
- **Markt-Regime:** Modell wurde über lange Zeit trainiert, Regimewechsel nicht explizit modelliert.
- **Daten:** Nur Preis/Volumen, keine Fundamentaldaten oder Sentiment.
