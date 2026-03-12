# multi_ticker_param_recommend.py
# -*- coding: utf-8 -*-
import gc
import os
import sys
import numpy as np
import pandas as pd
import vectorbt as vbt
import yfinance as yf
from tqdm import tqdm

# -------------------- 使用者參數 (可調) --------------------
TICKERS = ["AAPL","MSFT","AVGO","NVDA","GOOG","VOO","QQQ","TSLA","NFLX"]
START_DATE = "2024-01-01"
FREQ = "1d"

MA_LENS    = [10, 15, 20, 30, 50]
ATR_MULTS  = [1.5, 2.0, 2.5, 3.0]
ATR_LEN    = 14
LONG_MA_LEN = 200
USE_LONG_MA_FILTER = True

INIT_CASH       = 800
COMMISSION      = 0.002
SIZE_HACK_VALUE = 100000

OUT_DIR       = "multi_param_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
RECOMMEND_CSV = os.path.join(OUT_DIR, "recommendations.csv")

# -------------------- 小工具函式 --------------------
def safe_download_series(ticker, start, interval):
    try:
        data = vbt.YFData.download(ticker, start=start, interval=interval)
    except Exception:
        try:
            data = yf.download(ticker, start=start, interval=interval, progress=False)
        except Exception:
            return None, None, None, None

    def get_field(obj, key):
        if obj is None:
            return None
        if hasattr(obj, "get"):
            try:
                return obj.get(key)
            except Exception:
                pass
        if isinstance(obj, pd.DataFrame) and key in obj.columns:
            return obj[key]
        return None

    open_s  = get_field(data, "Open")
    high_s  = get_field(data, "High")
    low_s   = get_field(data, "Low")
    close_s = get_field(data, "Close")
    if close_s is None and isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
        close_s = data["Adj Close"]
    if close_s is None:
        return None, None, None, None

    close_s = pd.Series(close_s).dropna()
    if open_s is not None: open_s = pd.Series(open_s).reindex(close_s.index)
    if high_s is not None: high_s = pd.Series(high_s).reindex(close_s.index)
    if low_s  is not None: low_s  = pd.Series(low_s ).reindex(close_s.index)
    return open_s, high_s, low_s, close_s


def try_attr(obj, names):
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    try:
        return pd.Series(obj)
    except Exception as e:
        raise AttributeError(f"Indicator object missing {names}. Error: {e}")


def compute_trail_numpy(close_arr, atr_arr, entries_arr, atr_mult):
    n = len(close_arr)
    trail = np.full(n, np.nan)
    exits = np.zeros(n, dtype=bool)
    in_pos, cur_trail = False, np.nan
    for i in range(1, n):
        if np.isnan(atr_arr[i]):
            continue
        if entries_arr[i] and not in_pos:
            in_pos = True
            cur_trail = close_arr[i] - atr_arr[i] * atr_mult
            trail[i] = cur_trail
        elif in_pos:
            cur_trail = max(cur_trail, close_arr[i] - atr_arr[i] * atr_mult)
            trail[i] = cur_trail
            if close_arr[i] < cur_trail:
                exits[i] = True
                in_pos, cur_trail = False, np.nan
    return trail, exits


def compute_in_position(entries, exits):
    in_pos, last_entry_idx = False, None
    for idx in entries.index:
        if entries.loc[idx] and not in_pos:
            in_pos, last_entry_idx = True, idx
        if exits.loc[idx] and in_pos:
            in_pos, last_entry_idx = False, None
    return in_pos, last_entry_idx


# -------------------- 主流程：param grid --------------------
print("Python:", sys.version.splitlines()[0])
print("vectorbt:", getattr(vbt, "__version__", "unknown"))

all_results = []

for ticker in TICKERS:
    print(f"\n=== Processing {ticker} ===")
    open_s, high_s, low_s, close = safe_download_series(ticker, START_DATE, FREQ)
    if close is None or len(close) < 60:
        print(f"Skip {ticker}: no data or too short.")
        continue

    atr_run = vbt.ATR.run(high_s, low_s, close, window=ATR_LEN)
    atr = try_attr(atr_run, ["atr", "real", "value", "output"])
    if isinstance(atr, pd.DataFrame): atr = atr.iloc[:, 0]
    atr = atr.reindex(close.index)

    if USE_LONG_MA_FILTER:
        long_ma_run = vbt.MA.run(close, window=LONG_MA_LEN, ewm=False)
        long_ma = try_attr(long_ma_run, ["ma", "real", "value", "output"])
        if isinstance(long_ma, pd.DataFrame): long_ma = long_ma.iloc[:, 0]
        long_ma = long_ma.reindex(close.index)
    else:
        long_ma = pd.Series(np.nan, index=close.index)

    ma_run = vbt.MA.run(close, window=MA_LENS, ewm=False)
    ma_all = try_attr(ma_run, ["ma", "real", "value", "output"])
    if isinstance(ma_all, pd.Series):
        ma_all = ma_all.to_frame(name=str(MA_LENS[0]))
    try:
        if len(ma_all.columns) == len(MA_LENS):
            ma_all.columns = [str(int(w)) for w in MA_LENS]
    except Exception:
        pass

    close_df = pd.DataFrame(
        np.repeat(close.values[:, None], ma_all.shape[1], axis=1),
        index=close.index, columns=ma_all.columns
    )
    entries_df = (close_df > ma_all) & (close_df.shift(1) <= ma_all.shift(1))
    entries_df = entries_df.fillna(False).astype(bool)
    if USE_LONG_MA_FILTER:
        long_df = pd.DataFrame(
            np.repeat(long_ma.values[:, None], ma_all.shape[1], axis=1),
            index=close.index, columns=ma_all.columns
        )
        entries_df &= (close_df > long_df)
        del long_df

    close_arr = close.values
    atr_arr   = atr.values

    for atr_mult in tqdm(ATR_MULTS, desc=f"{ticker} ATRs"):
        for col in ma_all.columns:
            entries_arr = entries_df[col].values
            trail_arr, exits_arr = compute_trail_numpy(close_arr, atr_arr, entries_arr, atr_mult)

            pf = vbt.Portfolio.from_signals(
                close=close,
                entries=pd.Series(entries_arr, index=close.index),
                exits=pd.Series(exits_arr,     index=close.index),
                init_cash=INIT_CASH,
                fees=COMMISSION,
                size=SIZE_HACK_VALUE,
                direction="longonly",
                freq=FREQ
            )
            stats = pf.stats()
            pfactor      = stats.get("Profit Factor")    if "Profit Factor"    in stats.index else np.nan
            total_return = stats.get("Total Return [%]") if "Total Return [%]" in stats.index else np.nan
            max_dd       = stats.get("Max Drawdown [%]") if "Max Drawdown [%]" in stats.index else np.nan
            n_trades     = stats.get("Total Trades")     if "Total Trades"     in stats.index else 0
            sharpe       = stats.get("Sharpe ratio")     if "Sharpe ratio"     in stats.index else np.nan

            try:
                ma_len_val = int(''.join(filter(str.isdigit, str(col))))
            except Exception:
                ma_len_val = col

            all_results.append({
                'ticker':        ticker,
                'ma_len':        ma_len_val,
                'atr_mult':      float(atr_mult),
                'profit_factor': float(pfactor)      if not pd.isna(pfactor)      else np.nan,
                'total_return':  float(total_return) if not pd.isna(total_return) else np.nan,
                'max_drawdown':  float(max_dd)       if not pd.isna(max_dd)       else np.nan,
                'n_trades':      int(n_trades)        if not pd.isna(n_trades)     else 0,
                'sharpe':        float(sharpe)        if not pd.isna(sharpe)       else np.nan,
            })
            del pf

    del entries_df, ma_all, atr, open_s, high_s, low_s, close, close_arr, atr_arr
    gc.collect()

# -------------------- 每檔最佳組合 --------------------
res_df = pd.DataFrame(all_results)

best_list = []
for ticker in res_df['ticker'].unique():
    sub = res_df[res_df['ticker'] == ticker].sort_values("profit_factor", ascending=False).reset_index(drop=True)
    if not sub.empty:
        best_list.append(sub.iloc[0].to_dict())
best_df = pd.DataFrame(best_list)

# -------------------- 下一日建議 --------------------
recommendations = []

for _, row in best_df.iterrows():
    ticker   = row['ticker']
    best_ma  = int(row['ma_len'])
    best_atr = float(row['atr_mult'])
    print(f"\n--- Recommend {ticker}: MA={best_ma}, ATRx{best_atr} ---")

    _, high_s, low_s, close = safe_download_series(ticker, START_DATE, FREQ)
    if close is None or len(close) < 60:
        continue

    atr_run = vbt.ATR.run(high_s, low_s, close, window=ATR_LEN)
    atr = try_attr(atr_run, ["atr", "real", "value", "output"])
    if isinstance(atr, pd.DataFrame): atr = atr.iloc[:, 0]
    atr = atr.reindex(close.index)

    ma_run = vbt.MA.run(close, window=best_ma, ewm=False)
    ma_series = try_attr(ma_run, ["ma", "real", "value", "output"])
    if isinstance(ma_series, pd.DataFrame): ma_series = ma_series.iloc[:, 0]
    ma_series = ma_series.reindex(close.index)

    if USE_LONG_MA_FILTER:
        long_ma_run = vbt.MA.run(close, window=LONG_MA_LEN, ewm=False)
        long_ma = try_attr(long_ma_run, ["ma", "real", "value", "output"])
        if isinstance(long_ma, pd.DataFrame): long_ma = long_ma.iloc[:, 0]
        long_ma = long_ma.reindex(close.index)
    else:
        long_ma = pd.Series(np.nan, index=close.index)

    entries = (close > ma_series) & (close.shift(1) <= ma_series.shift(1))
    entries = entries.fillna(False).astype(bool)
    if USE_LONG_MA_FILTER:
        entries &= (close > long_ma)

    trail_arr, exits_arr = compute_trail_numpy(close.values, atr.values, entries.values, best_atr)
    exits = pd.Series(exits_arr, index=close.index)
    trail = pd.Series(trail_arr, index=close.index)

    in_pos_now, last_entry = compute_in_position(entries, exits)
    last_close   = float(close.iloc[-1])
    ma_last      = float(ma_series.iloc[-1]) if not pd.isna(ma_series.iloc[-1]) else None
    long_ma_last = float(long_ma.iloc[-1])   if not pd.isna(long_ma.iloc[-1])   else None
    last_trail   = float(trail.dropna().iloc[-1]) if trail.dropna().shape[0] > 0 else None

    if in_pos_now:
        action           = "HOLD"
        stop_or_rule     = f"Stop: {last_trail:.2f}" if last_trail else "—"
    elif entries.iloc[-1]:
        action           = "BUY"
        stop_or_rule     = f"Stop: {(close.iloc[-1] - atr.iloc[-1] * best_atr):.2f}" if not pd.isna(atr.iloc[-1]) else "—"
    else:
        action       = "WAIT"
        cond = f"Buy if close > {ma_last:.2f}" if ma_last else ""
        if long_ma_last:
            cond += f" & > LongMA {long_ma_last:.2f}"
        stop_or_rule = cond or "—"

    recommendations.append({
        "Ticker":         ticker,
        "Action":         action,
        "Last Close":     round(last_close, 2),
        "MA":             best_ma,
        "ATR x":          best_atr,
        "Profit Factor":  round(row['profit_factor'], 2) if not pd.isna(row['profit_factor']) else "—",
        "Stop / Rule":    stop_or_rule,
        "In Position":    "Yes" if in_pos_now else "No",
        "Last Entry":     str(last_entry.date()) if last_entry else "—",
    })

# -------------------- 儲存並印出 --------------------
rec_df = pd.DataFrame(recommendations)
rec_df.to_csv(RECOMMEND_CSV, index=False)

print("\n" + "="*60)
print(rec_df.to_string(index=False))
print("="*60)
print(f"\nSaved -> {RECOMMEND_CSV}")
print("Done.")
