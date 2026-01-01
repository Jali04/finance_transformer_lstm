import sys
import os

print("Python executable:", sys.executable)
print("Python version:", sys.version)

try:
    import numpy
    print("numpy:", numpy.__version__)
except ImportError as e:
    print("numpy error:", e)

try:
    import pandas
    print("pandas:", pandas.__version__)
except ImportError as e:
    print("pandas error:", e)

try:
    import matplotlib
    print("matplotlib:", matplotlib.__version__)
except ImportError as e:
    print("matplotlib error:", e)

try:
    import yfinance
    print("yfinance:", yfinance.__version__)
except ImportError as e:
    print("yfinance error:", e)

try:
    import tensorflow
    print("tensorflow:", tensorflow.__version__)
except ImportError as e:
    print("tensorflow error:", e)

# Check finance_lstm import
try:
    # Add parent dir to path as the notebook does
    ROOT = os.path.abspath("..")
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    
    import finance_lstm
    from finance_lstm import read_prices
    print("finance_lstm imported successfully")
except ImportError as e:
    print("finance_lstm error:", e)
except Exception as e:
    print("finance_lstm other error:", e)
