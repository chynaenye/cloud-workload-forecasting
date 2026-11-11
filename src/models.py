"""Forecasting models. Each takes (history: pd.Series, h: int) and returns h predictions."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

LAGS = [1, 2, 3, 6]


def persistence(history, h):
    """Future equals the last known value."""
    return np.repeat(history.iloc[-1], h)


def moving_average(history, h, window=6):
    """Future equals the mean of the last `window` values."""
    return np.repeat(history.iloc[-window:].mean(), h)


def make_features(series):
    """Lag features built only from past values."""
    df = pd.DataFrame({"y": series})
    for lag in LAGS:
        df[f"lag_{lag}"] = series.shift(lag)
    df["roll_mean_6"] = series.shift(1).rolling(6).mean()
    return df.dropna()


def lag_model(history, h):
    """Ridge regression on lags, forecasting one step at a time."""
    data = make_features(history)
    X, y = data.drop(columns="y"), data["y"]
    model = Ridge(alpha=1.0).fit(X, y)

    values = list(history.values)
    preds = []
    for _ in range(h):
        row = {f"lag_{lag}": values[-lag] for lag in LAGS}
        row["roll_mean_6"] = np.mean(values[-6:])
        p = model.predict(pd.DataFrame([row])[X.columns])[0]
        preds.append(p)
        values.append(p)  # feed the forecast back in
    return np.array(preds)


def ses_forecast(history, h):
    """Simple exponential smoothing."""
    history = history.reset_index(drop=True)  # statsmodels needs a plain index
    fit = SimpleExpSmoothing(history, initialization_method="estimated").fit(optimized=True)
    return fit.forecast(h).values


def clip_cpu(pred):
    return np.clip(pred, 0, 100)


def clip_count(pred):
    return np.maximum(pred, 0)


MODELS = {
    "persistence": persistence,
    "moving_average": moving_average,
    "lag_model": lag_model,
    "simple_exponential_smoothing": ses_forecast,
}

# Final choice per target (picked from rolling-origin validation on the training window)
FINAL_MODELS = {
    "active_vm_count": persistence,
    "aggregate_cpu_utilization": lag_model,
}