# multi_ticker_param_recommend.py
# -*- coding: utf-8 -*-
"""
策略版本：Walk-Forward Optimization (WFO)
- 滾動訓練窗口，參數每次自動更新，不依賴固定 Train/Test 切割
- WFO 統計跨窗口平均表現，用來評估策略是否真的有效
- 最後一個訓練窗口找出的參數 → 今日建議
- SPY 市場狀態過濾、相對強度輪動、量能確認、時間止損
"""
import gc
import os
import sys
import numpy as np
import pandas as pd
import vectorbt as vbt
import yfinance as yf
from tqdm import tqdm
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ================================================================
#  使用者參數
# ================================================================

UNIVERSE = {
    "個股": ["AAPL", "MSFT", "NVDA", "AVGO", "GOOG", "META", "AMZN", "TSLA", "NFLX"],
    "板塊": ["QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU"],
    "防禦": ["GLD", "TLT", "AGG", "VNQ"],
}
TICKERS    = [t for g in UNIVERSE.values() for t in g]
SPY_TICKER = "SPY"

# --- Walk-Forward 設定 ---
WFO_START        = "2023-01-01"   # 整個回測起點（要夠長才有足夠窗口）
WFO_TRAIN_MONTHS = 12             # 每個訓練窗口長度（月）
WFO_TEST_MONTHS  = 3              # 每個測試窗口長度（月）
# 今日建議用的訓練窗口 = 最近 WFO_TRAIN_MONTHS 個月

# --- 策略參數 grid ---
MA_LENS      = [10, 15, 20, 30, 50]
ATR_MULTS    = [1.5, 2.0, 2.5, 3.0]
ATR_LEN      = 14
LONG_MA_LEN  = 200

# --- 輪動設定 ---
MOMENTUM_WINDOW   = 60
TOP_N_TICKERS     = 5
MIN_AVG_TEST_PF   = 1.2   # WFO 各窗口平均 Profit Factor 門檻

# --- 出場 ---
MAX_HOLD_DAYS = 30

# --- 回測參數 ---
INIT_CASH       = 800
COMMISSION      = 0.002
SIZE_HACK_VALUE = 100000

# --- 輸出 ---
OUT_DIR        = "multi_param_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
RECOMMEND_CSV  = os.path.join(OUT_DIR, "recommendations.csv")
WFO_CSV        = os.path.join(OUT_DIR, "wfo_summary.csv")
HTML_BODY_FILE = os.path.join(OUT_DIR, "email_body.html")


# ================================================================
#  工具函式
# ================================================================

def safe_download_series(ticker, start, end=None):
    kwargs = {"start": start, "interval": "1d", "progress": False}
    if end:
        kwargs["end"] = end
    try:
        data = vbt.YFData.download(
            ticker, start=start, **( {"end": end} if end else {} ), interval="1d"
        )
    except Exception:
        data = None
    if data is None:
        try:
            data = yf.download(ticker, **kwargs)
        except Exception:
            return None, None, None, None

    def get_field(obj, key):
        if hasattr(obj, "get"):
            try:
                r = obj.get(key)
                if r is not None:
                    return r
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


def safe_download_volume(ticker, start, end=None):
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
        if "Volume" in df.columns:
            return pd.Series(df["Volume"]).dropna()
    except Exception:
        pass
    return None


def try_attr(obj, names):
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    try:
        return pd.Series(obj)
    except Exception as e:
        raise AttributeError(f"Indicator object missing {names}: {e}")


def compute_trail_numpy(close_arr, atr_arr, entries_arr, atr_mult, max_hold_days=None):
    n = len(close_arr)
    trail = np.full(n, np.nan)
    exits = np.zeros(n, dtype=bool)
    in_pos, cur_trail, entry_day, peak = False, np.nan, 0, np.nan

    for i in range(1, n):
        if np.isnan(atr_arr[i]):
            continue
        if entries_arr[i] and not in_pos:
            in_pos, entry_day, peak = True, i, close_arr[i]
            cur_trail = close_arr[i] - atr_arr[i] * atr_mult
            trail[i]  = cur_trail
        elif in_pos:
            if close_arr[i] > peak:
                peak = close_arr[i]
            cur_trail = max(cur_trail, close_arr[i] - atr_arr[i] * atr_mult)
            trail[i]  = cur_trail
            # 出場 1: ATR trailing stop
            if close_arr[i] < cur_trail:
                exits[i] = True
                in_pos, cur_trail = False, np.nan
                continue
            # 出場 2: 時間止損（超過 max_hold_days 且未明顯創新高）
            if max_hold_days and (i - entry_day) >= max_hold_days:
                if close_arr[i] <= peak * 0.995:
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


def get_market_regime(spy_close):
    if spy_close is None or len(spy_close) < LONG_MA_LEN:
        return "unknown"
    ma200      = spy_close.rolling(LONG_MA_LEN).mean()
    last_ma200 = ma200.iloc[-1]
    if pd.isna(last_ma200):
        return "unknown"
    return "risk-on" if spy_close.iloc[-1] > last_ma200 else "risk-off"


def compute_momentum_score(close, window=60):
    if close is None or len(close) < window + 1:
        return np.nan
    return (close.iloc[-1] - close.iloc[-window]) / close.iloc[-window]


def run_param_grid(ticker, close, high_s, low_s, ma_lens, atr_mults):
    """
    對已載入的 OHLC 跑完整 param grid，回傳 DataFrame
    """
    if close is None or len(close) < 60:
        return pd.DataFrame()

    atr_run = vbt.ATR.run(high_s, low_s, close, window=ATR_LEN)
    atr = try_attr(atr_run, ["atr", "real", "value", "output"])
    if isinstance(atr, pd.DataFrame): atr = atr.iloc[:, 0]
    atr = atr.reindex(close.index)

    long_ma_run = vbt.MA.run(close, window=LONG_MA_LEN, ewm=False)
    long_ma = try_attr(long_ma_run, ["ma", "real", "value", "output"])
    if isinstance(long_ma, pd.DataFrame): long_ma = long_ma.iloc[:, 0]
    long_ma = long_ma.reindex(close.index)

    ma_run = vbt.MA.run(close, window=ma_lens, ewm=False)
    ma_all = try_attr(ma_run, ["ma", "real", "value", "output"])
    if isinstance(ma_all, pd.Series):
        ma_all = ma_all.to_frame(name=str(ma_lens[0]))
    try:
        if len(ma_all.columns) == len(ma_lens):
            ma_all.columns = [str(int(w)) for w in ma_lens]
    except Exception:
        pass

    close_df = pd.DataFrame(
        np.repeat(close.values[:, None], ma_all.shape[1], axis=1),
        index=close.index, columns=ma_all.columns
    )
    entries_df = (close_df > ma_all) & (close_df.shift(1) <= ma_all.shift(1))
    entries_df = entries_df.fillna(False).astype(bool)
    long_df = pd.DataFrame(
        np.repeat(long_ma.values[:, None], ma_all.shape[1], axis=1),
        index=close.index, columns=ma_all.columns
    )
    entries_df &= (close_df > long_df)
    del long_df

    close_arr = close.values
    atr_arr   = atr.values
    results   = []

    for atr_mult in atr_mults:
        for col in ma_all.columns:
            entries_arr = entries_df[col].values
            trail_arr, exits_arr = compute_trail_numpy(
                close_arr, atr_arr, entries_arr, atr_mult, MAX_HOLD_DAYS
            )
            pf = vbt.Portfolio.from_signals(
                close=close,
                entries=pd.Series(entries_arr, index=close.index),
                exits=pd.Series(exits_arr,     index=close.index),
                init_cash=INIT_CASH, fees=COMMISSION,
                size=SIZE_HACK_VALUE, direction="longonly", freq="1d"
            )
            stats    = pf.stats()
            pfactor  = stats.get("Profit Factor")    if "Profit Factor"    in stats.index else np.nan
            ret      = stats.get("Total Return [%]") if "Total Return [%]" in stats.index else np.nan
            max_dd   = stats.get("Max Drawdown [%]") if "Max Drawdown [%]" in stats.index else np.nan
            n_trades = stats.get("Total Trades")     if "Total Trades"     in stats.index else 0
            sharpe   = stats.get("Sharpe ratio")     if "Sharpe ratio"     in stats.index else np.nan
            del pf

            try:
                ma_len_val = int(''.join(filter(str.isdigit, str(col))))
            except Exception:
                ma_len_val = col

            results.append({
                "ticker":        ticker,
                "ma_len":        ma_len_val,
                "atr_mult":      float(atr_mult),
                "profit_factor": float(pfactor)  if not pd.isna(pfactor)  else np.nan,
                "total_return":  float(ret)       if not pd.isna(ret)       else np.nan,
                "max_drawdown":  float(max_dd)    if not pd.isna(max_dd)    else np.nan,
                "n_trades":      int(n_trades)     if not pd.isna(n_trades)  else 0,
                "sharpe":        float(sharpe)    if not pd.isna(sharpe)    else np.nan,
            })

    del entries_df, ma_all, atr, close_arr, atr_arr
    gc.collect()
    return pd.DataFrame(results)


def generate_wfo_windows(wfo_start, train_months, test_months):
    """
    產生所有 [train_start, train_end, test_start, test_end] 窗口
    最後一個窗口的 test_end = today（用於今日建議）
    """
    today      = datetime.today().date()
    start_date = datetime.strptime(wfo_start, "%Y-%m-%d").date()
    windows    = []
    cursor     = start_date

    while True:
        train_start = cursor
        train_end   = cursor + relativedelta(months=train_months)
        test_start  = train_end
        test_end    = test_start + relativedelta(months=test_months)

        if test_end > today:
            # 最後一個窗口：test_end 設為 today
            windows.append((
                train_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
                test_start.strftime("%Y-%m-%d"),
                None   # None = 今天
            ))
            break

        windows.append((
            train_start.strftime("%Y-%m-%d"),
            train_end.strftime("%Y-%m-%d"),
            test_start.strftime("%Y-%m-%d"),
            test_end.strftime("%Y-%m-%d"),
        ))
        cursor += relativedelta(months=test_months)

    return windows


# ================================================================
#  主流程
# ================================================================
print("Python:", sys.version.splitlines()[0])
print("vectorbt:", getattr(vbt, "__version__", "unknown"))
print(f"Universe: {len(TICKERS)} tickers")

# ----------------------------------------------------------------
# Step 1: 市場狀態
# ----------------------------------------------------------------
print("\n[Step 1] Market regime...")
_, _, _, spy_close = safe_download_series(SPY_TICKER, WFO_START)
market_regime  = get_market_regime(spy_close)
ACTIVE_TICKERS = UNIVERSE["防禦"] if market_regime == "risk-off" else TICKERS
print(f"  Regime: {market_regime.upper()}  |  Active tickers: {len(ACTIVE_TICKERS)}")

# ----------------------------------------------------------------
# Step 2: 相對動能，選 Top N
# ----------------------------------------------------------------
print(f"\n[Step 2] Momentum ranking (window={MOMENTUM_WINDOW}d)...")
momentum_scores = {}
for ticker in ACTIVE_TICKERS:
    _, _, _, close = safe_download_series(ticker,
                         (datetime.today() - relativedelta(months=6)).strftime("%Y-%m-%d"))
    momentum_scores[ticker] = compute_momentum_score(close, MOMENTUM_WINDOW)
    gc.collect()

sorted_tickers = sorted(
    [(t, s) for t, s in momentum_scores.items() if not pd.isna(s)],
    key=lambda x: x[1], reverse=True
)
top_tickers = [t for t, _ in sorted_tickers[:TOP_N_TICKERS]]
print(f"  Top {TOP_N_TICKERS}: {top_tickers}")
for t in top_tickers:
    s = momentum_scores[t]
    print(f"    {t}: {s:.1%}")

# ----------------------------------------------------------------
# Step 3: 產生 WFO 窗口
# ----------------------------------------------------------------
windows = generate_wfo_windows(WFO_START, WFO_TRAIN_MONTHS, WFO_TEST_MONTHS)
print(f"\n[Step 3] WFO windows: {len(windows)} total  "
      f"(train={WFO_TRAIN_MONTHS}m, test={WFO_TEST_MONTHS}m)")
for i, (ts, te, vs, ve) in enumerate(windows):
    flag = " ← LIVE (今日建議用)" if ve is None else ""
    print(f"  Window {i+1:02d}: train {ts}~{te}  test {vs}~{ve or 'today'}{flag}")

# ----------------------------------------------------------------
# Step 4: 對每個 ticker 跑 WFO
# ----------------------------------------------------------------
print(f"\n[Step 4] Running WFO for top {TOP_N_TICKERS} tickers...")

# wfo_records: 儲存所有窗口的 test 結果，用於統計
wfo_records = []

# live_params: 最後一個窗口的最佳參數（今日建議用）
live_params = {}

for ticker in tqdm(top_tickers, desc="Tickers"):
    # 一次下載整段資料，各窗口切片使用，減少 API 呼叫
    _, high_full, low_full, close_full = safe_download_series(ticker, WFO_START)
    if close_full is None or len(close_full) < 100:
        print(f"  Skip {ticker}: no data")
        continue

    ticker_window_results = []

    for win_idx, (train_s, train_e, test_s, test_e) in enumerate(windows):
        is_live = (test_e is None)

        # 切片
        train_close = close_full.loc[train_s:train_e]
        train_high  = high_full.loc[train_s:train_e]  if high_full  is not None else None
        train_low   = low_full.loc[train_s:train_e]   if low_full   is not None else None

        if len(train_close) < 60:
            continue

        # 訓練：找最佳參數
        train_df = run_param_grid(ticker, train_close, train_high, train_low,
                                  MA_LENS, ATR_MULTS)
        if train_df.empty:
            continue

        best_row = train_df.sort_values("profit_factor", ascending=False).iloc[0]
        best_ma  = int(best_row["ma_len"])
        best_atr = float(best_row["atr_mult"])

        # 測試：用最佳參數跑 test 期間
        test_close = close_full.loc[test_s:]       if is_live else close_full.loc[test_s:test_e]
        test_high  = high_full.loc[test_s:]        if is_live else high_full.loc[test_s:test_e]
        test_low   = low_full.loc[test_s:]         if is_live else low_full.loc[test_s:test_e]

        if len(test_close) < 10:
            continue

        test_df = run_param_grid(ticker, test_close, test_high, test_low,
                                 [best_ma], [best_atr])
        if test_df.empty:
            continue

        test_pf  = float(test_df.iloc[0]["profit_factor"])
        test_ret = float(test_df.iloc[0]["total_return"])

        wfo_records.append({
            "ticker":    ticker,
            "window":    win_idx + 1,
            "is_live":   is_live,
            "train_s":   train_s,
            "train_e":   train_e,
            "test_s":    test_s,
            "test_e":    test_e or "today",
            "best_ma":   best_ma,
            "best_atr":  best_atr,
            "train_pf":  float(best_row["profit_factor"]),
            "test_pf":   test_pf,
            "test_ret":  test_ret,
        })

        ticker_window_results.append(test_pf)

        # 最後一個窗口 → live params
        if is_live:
            live_params[ticker] = {
                "ma_len":   best_ma,
                "atr_mult": best_atr,
                "train_pf": float(best_row["profit_factor"]),
                "test_pf":  test_pf,
                # WFO 平均 test PF（包含歷史窗口）
                "avg_test_pf": float(np.nanmean(ticker_window_results)) if ticker_window_results else np.nan,
                "n_windows":   len(ticker_window_results),
            }

    del close_full, high_full, low_full
    gc.collect()

# ----------------------------------------------------------------
# Step 5: WFO 統計 & 過濾
# ----------------------------------------------------------------
wfo_df = pd.DataFrame(wfo_records)
wfo_df.to_csv(WFO_CSV, index=False)
print(f"\n[Step 5] WFO summary saved -> {WFO_CSV}")

print("\n  WFO avg test Profit Factor per ticker:")
qualified_tickers = []
for ticker, params in live_params.items():
    avg_pf   = params["avg_test_pf"]
    n_win    = params["n_windows"]
    passed   = not pd.isna(avg_pf) and avg_pf >= MIN_AVG_TEST_PF
    status   = "✅ PASS" if passed else "❌ FAIL"
    print(f"    {ticker}: avg PF={avg_pf:.2f} over {n_win} windows  {status}")
    if passed:
        qualified_tickers.append(ticker)

print(f"\n  Qualified: {qualified_tickers}")

# ----------------------------------------------------------------
# Step 6: 今日建議
# ----------------------------------------------------------------
print("\n[Step 6] Generating today's recommendations...")
recommendations = []
today_start = (datetime.today() - relativedelta(months=WFO_TRAIN_MONTHS + 1)).strftime("%Y-%m-%d")

for ticker in qualified_tickers:
    params   = live_params[ticker]
    best_ma  = params["ma_len"]
    best_atr = params["atr_mult"]

    _, high_s, low_s, close = safe_download_series(ticker, today_start)
    if close is None or len(close) < 60:
        continue

    volume   = safe_download_volume(ticker, today_start)
    vol_ma20 = volume.rolling(20).mean() if volume is not None else None

    atr_run = vbt.ATR.run(high_s, low_s, close, window=ATR_LEN)
    atr = try_attr(atr_run, ["atr", "real", "value", "output"])
    if isinstance(atr, pd.DataFrame): atr = atr.iloc[:, 0]
    atr = atr.reindex(close.index)

    ma_run = vbt.MA.run(close, window=best_ma, ewm=False)
    ma_series = try_attr(ma_run, ["ma", "real", "value", "output"])
    if isinstance(ma_series, pd.DataFrame): ma_series = ma_series.iloc[:, 0]
    ma_series = ma_series.reindex(close.index)

    long_ma_run = vbt.MA.run(close, window=LONG_MA_LEN, ewm=False)
    long_ma = try_attr(long_ma_run, ["ma", "real", "value", "output"])
    if isinstance(long_ma, pd.DataFrame): long_ma = long_ma.iloc[:, 0]
    long_ma = long_ma.reindex(close.index)

    entries = (close > ma_series) & (close.shift(1) <= ma_series.shift(1))
    entries = entries.fillna(False).astype(bool)
    entries &= (close > long_ma)

    if vol_ma20 is not None:
        vol_ma20    = vol_ma20.reindex(close.index)
        vol_confirm = (volume.reindex(close.index) > vol_ma20 * 1.2).fillna(False)
        entries    &= vol_confirm

    trail_arr, exits_arr = compute_trail_numpy(
        close.values, atr.values, entries.values, best_atr, MAX_HOLD_DAYS
    )
    exits = pd.Series(exits_arr, index=close.index)
    trail = pd.Series(trail_arr, index=close.index)

    in_pos_now, last_entry = compute_in_position(entries, exits)
    last_close   = float(close.iloc[-1])
    ma_last      = float(ma_series.iloc[-1]) if not pd.isna(ma_series.iloc[-1]) else None
    long_ma_last = float(long_ma.iloc[-1])   if not pd.isna(long_ma.iloc[-1])   else None
    last_trail   = float(trail.dropna().iloc[-1]) if trail.dropna().shape[0] > 0 else None
    momentum_pct = momentum_scores.get(ticker, np.nan)

    if in_pos_now:
        action       = "HOLD"
        stop_or_rule = f"Stop: {last_trail:.2f}" if last_trail else "—"
    elif entries.iloc[-1]:
        action       = "BUY"
        init_stop    = close.iloc[-1] - atr.iloc[-1] * best_atr
        stop_or_rule = f"Stop: {init_stop:.2f}" if not pd.isna(atr.iloc[-1]) else "—"
    else:
        action = "WAIT"
        cond   = f"Buy if close > {ma_last:.2f}" if ma_last else ""
        if long_ma_last:
            cond += f" & > LongMA {long_ma_last:.2f}"
        stop_or_rule = cond or "—"

    recommendations.append({
        "Ticker":        ticker,
        "Action":        action,
        "Last Close":    round(last_close, 2),
        "Momentum 60d":  f"{momentum_pct:.1%}" if not pd.isna(momentum_pct) else "—",
        "MA":            best_ma,
        "ATR x":         best_atr,
        "Avg WFO PF":    round(params["avg_test_pf"], 2),
        "Live Test PF":  round(params["test_pf"], 2),
        "# Windows":     params["n_windows"],
        "Stop / Rule":   stop_or_rule,
        "In Position":   "Yes" if in_pos_now else "No",
        "Last Entry":    str(last_entry.date()) if last_entry else "—",
        "Market":        market_regime,
    })

    del high_s, low_s, close, atr, ma_series, long_ma, entries, exits, trail
    gc.collect()

# ----------------------------------------------------------------
# Step 7: 儲存 + HTML 信件
# ----------------------------------------------------------------
rec_df = pd.DataFrame(recommendations)
action_order = {"BUY": 0, "HOLD": 1, "WAIT": 2}
rec_df["_sort"] = rec_df["Action"].map(action_order).fillna(3)
rec_df = rec_df.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
rec_df.to_csv(RECOMMEND_CSV, index=False)

print("\n" + "="*75)
print(rec_df.to_string(index=False))
print("="*75)
print(f"Market: {market_regime.upper()}")
print(f"Saved recommendations -> {RECOMMEND_CSV}")
print(f"Saved WFO summary     -> {WFO_CSV}")

# HTML
ACTION_COLOR = {"BUY": "#d4edda", "HOLD": "#fff3cd", "WAIT": "#f8f9fa"}
rows_html = ""
for r in recommendations:
    color = ACTION_COLOR.get(r["Action"], "#ffffff")
    rows_html += f"""
    <tr style="background:{color}">
      <td><b>{r['Ticker']}</b></td>
      <td><b>{r['Action']}</b></td>
      <td>{r['Last Close']}</td>
      <td>{r['Momentum 60d']}</td>
      <td>{r['MA']}</td>
      <td>{r['ATR x']}</td>
      <td>{r['Avg WFO PF']}</td>
      <td>{r['Live Test PF']}</td>
      <td>{r['# Windows']}</td>
      <td>{r['Stop / Rule']}</td>
      <td>{r['In Position']}</td>
      <td>{r['Last Entry']}</td>
    </tr>"""

regime_color = "#d4edda" if market_regime == "risk-on" else "#f8d7da"
today_str    = datetime.now().strftime("%Y-%m-%d")

html_body = f"""
<html><body style="font-family:Arial;padding:16px">
<h2>📈 Daily Trading Recommendations — {today_str}</h2>
<p>
  Market Regime:
  <span style="background:{regime_color};padding:3px 10px;border-radius:4px">
    <b>{market_regime.upper()}</b>
  </span>
  &nbsp;|&nbsp; WFO: train={WFO_TRAIN_MONTHS}m / test={WFO_TEST_MONTHS}m
  &nbsp;|&nbsp; Min avg WFO PF: {MIN_AVG_TEST_PF}
  &nbsp;|&nbsp; Top {TOP_N_TICKERS} by {MOMENTUM_WINDOW}d momentum
</p>
<table border="1" cellpadding="6" cellspacing="0"
       style="border-collapse:collapse;font-size:13px">
  <thead>
    <tr style="background:#343a40;color:white">
      <th>Ticker</th><th>Action</th><th>Last Close</th>
      <th>Momentum 60d</th><th>MA</th><th>ATR x</th>
      <th>Avg WFO PF</th><th>Live Test PF</th><th># Windows</th>
      <th>Stop / Rule</th><th>In Position</th><th>Last Entry</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
<p style="font-size:12px;color:gray;margin-top:12px">
  BUY=<span style="background:#d4edda;padding:2px 6px">green</span> &nbsp;
  HOLD=<span style="background:#fff3cd;padding:2px 6px">yellow</span> &nbsp;
  WAIT=<span style="background:#f8f9fa;padding:2px 6px">gray</span><br>
  <b>Avg WFO PF</b>: 所有歷史測試窗口的平均 Profit Factor（越高越穩定）<br>
  <b>Live Test PF</b>: 最近一個測試窗口的 Profit Factor<br>
  ⚠️ 僅供參考，不構成投資建議。過去績效不代表未來表現。
</p>
</body></html>
"""

with open(HTML_BODY_FILE, "w", encoding="utf-8") as f:
    f.write(html_body)

print(f"Saved HTML email      -> {HTML_BODY_FILE}")
print("Done.")
