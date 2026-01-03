# Block 7 – Abschluss-Report

- **Generiert:** 2026-01-03T20:14:43Z
- **Run:** `2026-01-03_21-02-35_lstm`
- **Modell-Setup:** Ticker `AAPL` | Horizon `1` | Lookback `60`
---
## 1. Klassifikations-Leistung (Test Set)
Wir bewerten, wie gut das Modell Wahrscheinlichkeiten vorhersagt.
- **AUROC**: `0.491` (Fläche unter ROC Kurve, 0.5 = Zufall)
- **AUPRC**: `0.526` (Fläche unter Precision-Recall Kurve, Basisrate ≈ 0.53)
- **MCC**: `-0.019` (Matthews Correlation Coefficient)

### Grafiken

![ROC](figures\roc_test.png)  
![PR](figures\pr_test.png)

## 2. Finanzielle Performance (Backtest)
Simuliertes Handelsergebnis mit **15.0** bps Roundtrip-Kosten und T+1 Entry:
- **CAGR**: `-15.90%` (Jährliche Rendite)
- **Sharpe Ratio**: `-1.32` (Risikoadjustierte Rendite)
- **Max Drawdown**: `-34.25%` (Maximaler zwischenzeitlicher Verlust)
- **Endkapital**: `0.69` (Start = 1.00)

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