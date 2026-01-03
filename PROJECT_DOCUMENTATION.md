
# Projekt-Dokumentation: Financial Forecasting mit LSTM & Transformer

## 1. ML Pipeline Steps
Unser Machine Learning Workflow ist modular aufgebaut und besteht aus folgenden Schritten:

1.  **Data Loading & Preprocessing (`01_02_data_load.ipynb`)**:
    *   Laden von Finanzdaten (z.B. OHLCV).
    *   Berechnung technischer Indikatoren (RSI, MACD, Bollinger Bands etc.).
    *   Erstellung des Zielvariable (Target Labeling) für Klassifikation (Steigt/Fällt).
2.  **Hyperparameter Search (`05_wfcv_search.ipynb`)**:
    *   Walk-Forward Cross-Validation (WFCV) zur robusten Suche nach besten Parametern ohne Lookahead-Bias.
    *   Optimierung von Modell-Parametern (Layers, Dropout, Learning Rate) und Feature-Sets.
3.  **Model Training (`03_train_lstm.ipynb` / `03_train_transformer.ipynb`)**:
    *   **LSTM**: Nutzung von Keras/TensorFlow. Sequentielle Verarbeitung mit GRU/LSTM Layern.
    *   **Transformer**: Nutzung von PyTorch. Custom `TimeSeriesTransformer` mit Multi-Head Attention und Positional Encoding.
    *   Beide Modelle nutzen denselben Scaler und dieselben Daten-Splits (Train/Val/Test).
4.  **Evaluation & Backtest (`04_eval_backtest.ipynb`)**:
    *   Berechnung von Metriken auf ungesehenen Testdaten (Accuracy, MCC, ROC-AUC).
    *   **XAI (Explainable AI)**: Integration von SHAP (Feature Importance) und Attention Maps (Transformer), um "Black Box" Entscheidungen nachvollziehbar zu machen.
    *   Finanzieller Backtest: Simulation einer Handelsstrategie basierend auf Modell-Signalen (Equity Curve).

## 2. Design Decisions
*   **Modell-Vergleich**: Wir setzen LSTM (Recurrent) gegen Transformer (Attention-based), um zu prüfen, ob der moderne Transformer-Ansatz auch bei kurzen Zeitreihen Vorteile bietet.
*   **Walk-Forward Validation**: Finanzdaten sind nicht i.i.d. (independent and identically distributed). Ein normaler K-Fold Cross-Validation würde Zukunftsinformationen leaken. Daher nutzen wir strikte Zeit-Fenster (Trainieren auf Vergangenheit, Testen auf direkter Zukunft).
*   **Feature Engineering**: Statt Rohdaten nutzen wir eine breite Palette technischer Indikatoren, reduzieren diese aber selektiv, um den "Curse of Dimensionality" zu vermeiden.
*   **XAI Integration**: Um Vertrauen in die Modelle zu schaffen, wählten wir SHAP als modell-agnostische Methode und Attention-Visualization speziell für den Transformer.

## 3. Experience / Self-reflection
*   **Herausforderung Overfitting**: Finanzdaten haben ein extrem niedriges Signal-zu-Rausch-Verhältnis. Modelle neigen dazu, Rauschen zu lernen. Strikte Regularisierung (Dropout, kleine Modelle) war essentiell.
*   **Datenqualität**: Fehlerhafte oder lückenhafte Daten können das Training massiv stören. Viel Zeit floss in das Cleaning und Alignment der Zeitstempel.
*   **Komplexität vs. Nutzen**: Der Transformer ist deutlich komplexer zu implementieren und zu trainieren als das LSTM. In ersten Tests zeigte sich, dass LSTMs auf solchen Tabellendaten oft robuster ("easier to tune") sind, während Transformer mehr Daten benötigen, um ihre Stärke auszuspielen.
*   **XAI Analyse Ergebnisse**: Die durchgeführte SHAP-Analyse (siehe `shap_summary.png`) zeigt eine starke Dominanz des Features `logret_1d` (Rendite des Vortages). Alle anderen technischen Indikatoren spielen eine vernachlässigbare Rolle. Zudem sind die absoluten SHAP-Werte sehr niedrig, was auf eine hohe Unsicherheit des Modells hinweist. Dies bestätigt die "Random Walk" Theorie und das niedrige Signal-zu-Rausch-Verhältnis: Das Modell verlässt sich hauptsächlich auf die jüngste Kursbewegung, findet aber kein starkes prädiktives Muster.

## 4. Outlook
*   **Mehr Daten**: Einbeziehung alternativer Daten (Sentiment Analysis, News) könnte das Signal verbessern.
*   **Ensemble Learning**: Kombination von LSTM und Transformer (Voting oder Stacking) könnte die Varianz der Vorhersagen reduzieren.
*   **Reinforcement Learning**: Statt nur Vorhersagen zu treffen ("Steigt es?"), könnte ein RL-Agent direkt Handelsentscheidungen ("Kaufen/Verkaufen") lernen und dabei Transaktionskosten berücksichtigen.
