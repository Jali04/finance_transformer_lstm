# Block 7 – Abschluss-Report

- **Generiert:** 2026-01-07T20:36:28Z
- **Run:** `2026-01-07_21-29-55_lstm`
- **Modell-Setup:** Ticker `AAPL` | Horizon `1` | Lookback `60`
---
## 1. Klassifikations-Leistung (Test Set)
Wir bewerten, wie gut das Modell Wahrscheinlichkeiten vorhersagt.
- **AUROC**: `0.499` (Fläche unter ROC Kurve, 0.5 = Zufall)
- **AUPRC**: `0.521` (Fläche unter Precision-Recall Kurve, Basisrate ≈ 0.53)
- **MCC**: `0.035` (Matthews Correlation Coefficient, >0 ist besser als Zufall)

### Grafiken

![ROC](figures\roc_test.png)  
![PR](figures\pr_test.png)

## 2. Finanzielle Performance (Backtest)
Simuliertes Handelsergebnis mit **15.0** bps Roundtrip-Kosten und T+1 Entry:
- **CAGR**: `21.07%` (Jährliche Rendite)
- **Sharpe Ratio**: `0.64` (Risikoadjustierte Rendite)
- **Max Drawdown**: `-22.99%` (Maximaler zwischenzeitlicher Verlust)
- **Endkapital**: `1.16` (Start = 1.00)

![Equity Trace](figures\equity_costed.png)

## 3. Vergleich mit Benchmarks
Ist das Modell besser als simple Methoden? (Lift Factor > 1.0)
| Baseline | AUPRC | Lift Factor |
|---|---|---|
| Always-Up (Prior) | 0.527 | 1.00x |
| Logistic Regression | 0.521 | 0.99x |
| Simple MACD | 0.517 | 0.98x |

---
**Ende des Berichts.**