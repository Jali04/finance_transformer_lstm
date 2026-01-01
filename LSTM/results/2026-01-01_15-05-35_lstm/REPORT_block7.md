# Block 7 – Abschluss-Report

- **Run-Ordner:** `2026-01-01_15-05-35_lstm`
- **Erstellt (UTC):** 2026-01-01T14:07:23Z
- **Ticker/Intervall:** AAPL / 1d
- **Zeitraum:** 2012-01-01 → 2025-09-01
- **Horizon/Lookback:** H=1 / LB=60
- **Featureset:** v2 → logret_1d, logret_3d, logret_5d, realized_vol_10, bb_pos, rsi_14, macd, macd_sig, macd_diff, vol_z_20, sma_diff

## Test-Metriken
- AUROC: **0.516**, AUPRC: **0.532** (Random=0.53? Check PosRate), Brier: **0.258**
- Balanced Acc: **0.533**, MCC: **0.067**

![ROC](figures\roc_test.png)  
![PR](figures\pr_test.png)

## Backtests
- **Gross (ohne Kosten)**: Entry@t CAGR=0.139, Entry@t+1 CAGR=0.105
## Kosten-KPI (realistisch)
- **Netto (mit Kosten)**: Entry@t+1 (No-Overlap) @ 15.0 bps. CAGR=-0.138, Equity=0.766
![Equity net](figures\equity_costed.png)

## Limitations
- **Labeling:** Sensitivität gegenüber Epsilon ist hoch.
- **Markt-Regime:** Modell wurde über lange Zeit trainiert, Regimewechsel nicht explizit modelliert.
- **Daten:** Nur Preis/Volumen, keine Fundamentaldaten oder Sentiment.
