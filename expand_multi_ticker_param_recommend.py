# multi_ticker_param_recommend.py
# -*- coding: utf-8 -*-
"""
中線波段策略（2~6週）
進場：50MA趨勢過濾 + 爆量突破 + K棒強度 三層確認
出場：ATR trailing stop + K線反轉訊號（任一先到即出場）
驗證：固定 Train/Test 切割
選股：21檔（個股+板塊ETF+防禦）+ SPY市場狀態 + 動能輪動前5
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

# --- 時間切割 ---
TRAIN_START = "2021-01-01"
TRAIN_END   = "2023-12-31"
TEST_START  = "2024-01-01"

# --- 進場參數 grid（train 期間掃這些組合）---
TREND_MA_LENS   = [30, 50]          # 趨勢均線
TREND_SLOPE_WIN = 10                # 均線斜率計算窗口（天）
VOL_MULT_LIST   = [1.3, 1.5, 2.0]  # 爆量門檻（相對20日均量倍數）
BODY_RATIO_LIST = [0.5, 0.6, 0.7]  # K棒實體比例門檻
CLOSE_TOP_LIST  = [0.20, 0.25]     # 收盤在當天高點前 N% 以內

# --- 出場參數 ---
ATR_LEN       = 14
ATR_MULT_LIST = [1.5, 2.0, 2.5]    # ATR trailing stop 倍數
MAX_HOLD_DAYS = 30                  # 時間止損

# K線反轉：射擊之星 / 吞噬
UPPER_SHADOW_RATIO = 2.0   # 上影線 > 實體 N 倍 → 反轉訊號
ENGULF_RATIO       = 1.0   # 陰線實體 > 前陽線實體 N 倍 → 吞噬
WEAK_VOL_DAYS      = 3     # 連續縮量陰線天數

# --- 輪動 ---
MOMENTUM_WINDOW = 60
TOP_N_TICKERS   = 5

# --- 驗證門檻 ---
MIN_TEST_PF   = 1.2
MIN_TRADES    = 3     # 測試期至少要有幾筆交易才算有效

# --- 回測 ---
INIT_CASH       = 800
COMMISSION      = 0.002
SIZE_HACK_VALUE = 100000

# --- 輸出 ---
OUT_DIR        = "multi_param_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
RECOMMEND_CSV  = os.path.join(OUT_DIR, "recommendations.csv")
HTML_BODY_FILE = os.path.join(OUT_DIR, "email_body.html")


# ================================================================
#  工具函式
# ================================================================

def safe_download(ticker, start, end=None):
    """下載 OHLCV，回傳 DataFrame 或 None"""
    kwargs = {"start": start, "interval": "1d", "progress": False}
    if end:
        kwargs["end"] = end
    try:
        df = yf.download(ticker, **kwargs)
        if df is None or df.empty:
            raise ValueError("empty")
        # 有時候 yfinance 回傳 MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None


def extract_ohlcv(df):
    """從 DataFrame 取出乾淨的 open/high/low/close/volume Series"""
    if df is None:
        return None, None, None, None, None
    def col(key):
        for k in [key, key.lower(), key.capitalize()]:
            if k in df.columns:
                s = df[k]
                return pd.Series(s.values, index=df.index).dropna() if not isinstance(s, pd.Series) else s.dropna()
        return None
    close  = col("Close") or col("Adj Close")
    if close is None:
        return None, None, None, None, None
    idx    = close.index
    open_s = col("Open");  open_s  = open_s.reindex(idx)  if open_s  is not None else pd.Series(np.nan, index=idx)
    high_s = col("High");  high_s  = high_s.reindex(idx)  if high_s  is not None else pd.Series(np.nan, index=idx)
    low_s  = col("Low");   low_s   = low_s.reindex(idx)   if low_s   is not None else pd.Series(np.nan, index=idx)
    vol    = col("Volume"); vol    = vol.reindex(idx)      if vol     is not None else pd.Series(np.nan, index=idx)
    return open_s, high_s, low_s, close, vol


def try_attr(obj, names):
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return pd.Series(obj)


# ================================================================
#  進場訊號計算
# ================================================================

def compute_entry_signals(open_s, high_s, low_s, close, vol,
                           trend_ma_len, vol_mult, body_ratio, close_top):
    """
    三層進場條件：
    1. 趨勢：close > trend_MA 且 MA 斜率向上
    2. 爆量突破：volume > vol_mult * vol_MA20
    3. K棒強度：實體 > body_ratio * 全棒，收在高點 close_top 以內，收紅
    """
    # 趨勢均線
    ma_run    = vbt.MA.run(close, window=trend_ma_len, ewm=False)
    trend_ma  = try_attr(ma_run, ["ma", "real", "value", "output"])
    if isinstance(trend_ma, pd.DataFrame): trend_ma = trend_ma.iloc[:, 0]
    trend_ma  = trend_ma.reindex(close.index)
    ma_slope  = trend_ma.diff(TREND_SLOPE_WIN)          # 正值 = 向上

    # 成交量均線
    vol_ma20  = vol.rolling(20).mean()

    # --- 條件 1: 趨勢 ---
    cond_trend = (close > trend_ma) & (ma_slope > 0)

    # --- 條件 2: 爆量 ---
    cond_vol   = vol > vol_mult * vol_ma20

    # --- 條件 3: K棒強度 ---
    full_range = high_s - low_s                         # 全棒長度
    body       = (close - open_s).abs()                 # 實體大小
    upper_wick = high_s - close.where(close >= open_s, open_s)  # 上影線
    cond_body  = (body / full_range.replace(0, np.nan)) >= body_ratio
    cond_top   = (high_s - close) / full_range.replace(0, np.nan) <= close_top
    cond_bull  = close > open_s                         # 收紅

    entries = (cond_trend & cond_vol & cond_body & cond_top & cond_bull)
    entries = entries.fillna(False).astype(bool)
    return entries


# ================================================================
#  出場訊號計算
# ================================================================

def compute_exit_signals(open_s, high_s, low_s, close, vol,
                          atr, atr_mult, entries):
    """
    雙重出場條件：
    1. ATR trailing stop
    2. K線反轉訊號（射擊之星 / 吞噬 / 縮量陰線連續）
    任一先到即出場
    """
    n          = len(close)
    close_arr  = close.values
    open_arr   = open_s.values
    high_arr   = high_s.values
    low_arr    = low_s.values
    vol_arr    = vol.values
    atr_arr    = atr.values
    entry_arr  = entries.values

    trail  = np.full(n, np.nan)
    exits  = np.zeros(n, dtype=bool)
    in_pos = False
    cur_trail, entry_day, peak = np.nan, 0, np.nan
    consec_weak = 0   # 連續縮量陰線計數

    vol_ma20 = pd.Series(vol_arr).rolling(20).mean().values

    for i in range(1, n):
        if np.isnan(atr_arr[i]):
            continue

        if entry_arr[i] and not in_pos:
            in_pos     = True
            entry_day  = i
            peak       = close_arr[i]
            cur_trail  = close_arr[i] - atr_arr[i] * atr_mult
            trail[i]   = cur_trail
            consec_weak = 0
            continue

        if in_pos:
            if close_arr[i] > peak:
                peak = close_arr[i]

            # 更新 ATR trailing stop
            cur_trail = max(cur_trail, close_arr[i] - atr_arr[i] * atr_mult)
            trail[i]  = cur_trail

            # --- 出場 1: ATR trailing stop ---
            if close_arr[i] < cur_trail:
                exits[i] = True
                in_pos, cur_trail = False, np.nan
                continue

            # --- 出場 2: 時間止損 ---
            if (i - entry_day) >= MAX_HOLD_DAYS and close_arr[i] <= peak * 0.995:
                exits[i] = True
                in_pos, cur_trail = False, np.nan
                continue

            # --- 出場 3: K線反轉 ---
            full   = high_arr[i] - low_arr[i]
            body   = abs(close_arr[i] - open_arr[i])
            upper  = high_arr[i] - max(close_arr[i], open_arr[i])

            # 射擊之星 / 墓碑：上影線 > 實體 N 倍，收在低位
            is_shooting_star = (
                full > 0 and body > 0 and
                upper >= UPPER_SHADOW_RATIO * body and
                (close_arr[i] - low_arr[i]) / full <= 0.35
            )

            # 吞噬陰線：今天陰線實體 > 昨天陽線實體
            prev_body = abs(close_arr[i-1] - open_arr[i-1])
            is_bearish_engulf = (
                close_arr[i] < open_arr[i] and           # 今收陰
                close_arr[i-1] > open_arr[i-1] and       # 昨收陽
                body >= ENGULF_RATIO * prev_body and      # 今實體 >= 昨實體
                close_arr[i] < open_arr[i-1]             # 今收 < 昨開
            )

            # 縮量陰線
            is_weak_bear = (
                close_arr[i] < open_arr[i] and
                not np.isnan(vol_ma20[i]) and
                vol_arr[i] < vol_ma20[i] * 0.8
            )
            consec_weak = (consec_weak + 1) if is_weak_bear else 0
            is_vol_exhaustion = (consec_weak >= WEAK_VOL_DAYS)

            if is_shooting_star or is_bearish_engulf or is_vol_exhaustion:
                exits[i] = True
                in_pos, cur_trail = False, np.nan
                consec_weak = 0
                continue

    return pd.Series(trail, index=close.index), pd.Series(exits, index=close.index)


# ================================================================
#  單一 ticker param grid
# ================================================================

def run_param_grid(ticker, open_s, high_s, low_s, close, vol):
    """跑完整 param grid，回傳 results list"""
    if close is None or len(close) < 80:
        return []

    atr_run = vbt.ATR.run(high_s, low_s, close, window=ATR_LEN)
    atr = try_attr(atr_run, ["atr", "real", "value", "output"])
    if isinstance(atr, pd.DataFrame): atr = atr.iloc[:, 0]
    atr = atr.reindex(close.index)

    results = []
    for trend_ma_len in TREND_MA_LENS:
        for vol_mult in VOL_MULT_LIST:
            for body_ratio in BODY_RATIO_LIST:
                for close_top in CLOSE_TOP_LIST:
                    entries = compute_entry_signals(
                        open_s, high_s, low_s, close, vol,
                        trend_ma_len, vol_mult, body_ratio, close_top
                    )
                    n_entry = entries.sum()
                    if n_entry == 0:
                        continue

                    for atr_mult in ATR_MULT_LIST:
                        _, exits = compute_exit_signals(
                            open_s, high_s, low_s, close, vol,
                            atr, atr_mult, entries
                        )
                        pf = vbt.Portfolio.from_signals(
                            close=close,
                            entries=entries,
                            exits=exits,
                            init_cash=INIT_CASH,
                            fees=COMMISSION,
                            size=SIZE_HACK_VALUE,
                            direction="longonly",
                            freq="1d"
                        )
                        stats    = pf.stats()
                        pfactor  = stats.get("Profit Factor")    if "Profit Factor"    in stats.index else np.nan
                        ret      = stats.get("Total Return [%]") if "Total Return [%]" in stats.index else np.nan
                        max_dd   = stats.get("Max Drawdown [%]") if "Max Drawdown [%]" in stats.index else np.nan
                        n_trades = stats.get("Total Trades")     if "Total Trades"     in stats.index else 0
                        sharpe   = stats.get("Sharpe ratio")     if "Sharpe ratio"     in stats.index else np.nan
                        del pf

                        results.append({
                            "ticker":       ticker,
                            "trend_ma":     trend_ma_len,
                            "vol_mult":     vol_mult,
                            "body_ratio":   body_ratio,
                            "close_top":    close_top,
                            "atr_mult":     atr_mult,
                            "profit_factor":float(pfactor)  if not pd.isna(pfactor)  else np.nan,
                            "total_return": float(ret)      if not pd.isna(ret)       else np.nan,
                            "max_drawdown": float(max_dd)   if not pd.isna(max_dd)    else np.nan,
                            "n_trades":     int(n_trades)   if not pd.isna(n_trades)  else 0,
                            "sharpe":       float(sharpe)   if not pd.isna(sharpe)    else np.nan,
                        })

    gc.collect()
    return results


def get_market_regime(spy_close):
    if spy_close is None or len(spy_close) < 200:
        return "unknown"
    ma200 = spy_close.rolling(200).mean()
    if pd.isna(ma200.iloc[-1]):
        return "unknown"
    return "risk-on" if spy_close.iloc[-1] > ma200.iloc[-1] else "risk-off"


def compute_momentum_score(close, window=60):
    if close is None or len(close) < window + 1:
        return np.nan
    return (close.iloc[-1] - close.iloc[-window]) / close.iloc[-window]


def compute_in_position(entries, exits):
    in_pos, last_entry_idx = False, None
    for idx in entries.index:
        if entries.loc[idx] and not in_pos:
            in_pos, last_entry_idx = True, idx
        if exits.loc[idx] and in_pos:
            in_pos, last_entry_idx = False, None
    return in_pos, last_entry_idx


# ================================================================
#  主流程
# ================================================================
print("Python:", sys.version.splitlines()[0])
print("vectorbt:", getattr(vbt, "__version__", "unknown"))
print(f"Universe: {len(TICKERS)} tickers")
print(f"Train: {TRAIN_START} ~ {TRAIN_END}  |  Test: {TEST_START} ~ today")

# ----------------------------------------------------------------
# Step 1: 市場狀態
# ----------------------------------------------------------------
print("\n[Step 1] Market regime...")
spy_df = safe_download(SPY_TICKER, TRAIN_START)
_, _, _, spy_close, _ = extract_ohlcv(spy_df)
market_regime  = get_market_regime(spy_close)
ACTIVE_TICKERS = UNIVERSE["防禦"] if market_regime == "risk-off" else TICKERS
print(f"  Regime: {market_regime.upper()}  |  Active: {len(ACTIVE_TICKERS)} tickers")

# ----------------------------------------------------------------
# Step 2: 動能排名，選 Top N
# ----------------------------------------------------------------
print(f"\n[Step 2] Momentum ranking (window={MOMENTUM_WINDOW}d)...")
momentum_scores = {}
for ticker in ACTIVE_TICKERS:
    df = safe_download(ticker, TEST_START)
    _, _, _, close, _ = extract_ohlcv(df)
    momentum_scores[ticker] = compute_momentum_score(close, MOMENTUM_WINDOW)
    del df
    gc.collect()

sorted_tickers = sorted(
    [(t, s) for t, s in momentum_scores.items() if not pd.isna(s)],
    key=lambda x: x[1], reverse=True
)
top_tickers = [t for t, _ in sorted_tickers[:TOP_N_TICKERS]]
print(f"  Top {TOP_N_TICKERS}: " + ", ".join(f"{t}({momentum_scores[t]:.1%})" for t in top_tickers))

# ----------------------------------------------------------------
# Step 3: Train param grid
# ----------------------------------------------------------------
print(f"\n[Step 3] Train param grid ({TRAIN_START} ~ {TRAIN_END})...")
train_all = []
for ticker in tqdm(top_tickers, desc="Train"):
    df = safe_download(ticker, TRAIN_START, TRAIN_END)
    open_s, high_s, low_s, close, vol = extract_ohlcv(df)
    if close is None or len(close) < 80:
        print(f"  Skip {ticker}: insufficient data")
        continue
    results = run_param_grid(ticker, open_s, high_s, low_s, close, vol)
    train_all.extend(results)
    del df, open_s, high_s, low_s, close, vol
    gc.collect()

train_df = pd.DataFrame(train_all)

# 每檔取最佳參數（PF 最高且交易次數 >= MIN_TRADES）
best_params = {}
for ticker in top_tickers:
    sub = train_df[
        (train_df["ticker"] == ticker) &
        (train_df["n_trades"] >= MIN_TRADES)
    ].sort_values("profit_factor", ascending=False)
    if sub.empty:
        print(f"  {ticker}: no valid combos in train (all < {MIN_TRADES} trades)")
        continue
    best = sub.iloc[0]
    best_params[ticker] = {
        "trend_ma":   int(best["trend_ma"]),
        "vol_mult":   float(best["vol_mult"]),
        "body_ratio": float(best["body_ratio"]),
        "close_top":  float(best["close_top"]),
        "atr_mult":   float(best["atr_mult"]),
        "train_pf":   float(best["profit_factor"]),
        "train_trades": int(best["n_trades"]),
    }
    print(f"  {ticker}: MA{best_params[ticker]['trend_ma']} "
          f"vol>{best_params[ticker]['vol_mult']}x "
          f"body>{best_params[ticker]['body_ratio']:.0%} "
          f"ATR×{best_params[ticker]['atr_mult']}  "
          f"→ Train PF={best_params[ticker]['train_pf']:.2f} "
          f"({best_params[ticker]['train_trades']} trades)")

# ----------------------------------------------------------------
# Step 4: Test 驗證
# ----------------------------------------------------------------
print(f"\n[Step 4] Test validation ({TEST_START} ~ today)...")
qualified = {}
for ticker, params in tqdm(best_params.items(), desc="Test"):
    df = safe_download(ticker, TEST_START)
    open_s, high_s, low_s, close, vol = extract_ohlcv(df)
    if close is None or len(close) < 30:
        continue

    atr_run = vbt.ATR.run(high_s, low_s, close, window=ATR_LEN)
    atr = try_attr(atr_run, ["atr", "real", "value", "output"])
    if isinstance(atr, pd.DataFrame): atr = atr.iloc[:, 0]
    atr = atr.reindex(close.index)

    entries = compute_entry_signals(
        open_s, high_s, low_s, close, vol,
        params["trend_ma"], params["vol_mult"],
        params["body_ratio"], params["close_top"]
    )
    _, exits = compute_exit_signals(
        open_s, high_s, low_s, close, vol,
        atr, params["atr_mult"], entries
    )

    pf    = vbt.Portfolio.from_signals(
        close=close, entries=entries, exits=exits,
        init_cash=INIT_CASH, fees=COMMISSION,
        size=SIZE_HACK_VALUE, direction="longonly", freq="1d"
    )
    stats    = pf.stats()
    test_pf  = float(stats.get("Profit Factor")    if "Profit Factor"    in stats.index else np.nan)
    test_ret = float(stats.get("Total Return [%]") if "Total Return [%]" in stats.index else np.nan)
    n_trades = int(  stats.get("Total Trades")     if "Total Trades"     in stats.index else 0)
    del pf

    passed = (not pd.isna(test_pf)) and test_pf >= MIN_TEST_PF and n_trades >= MIN_TRADES
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {ticker}: test PF={test_pf:.2f}, trades={n_trades}  {status}")

    if passed:
        params["test_pf"]     = test_pf
        params["test_return"] = test_ret
        params["test_trades"] = n_trades
        # 保留最新的指標物件用於今日建議
        params["_close"]   = close
        params["_open"]    = open_s
        params["_high"]    = high_s
        params["_low"]     = low_s
        params["_vol"]     = vol
        params["_atr"]     = atr
        params["_entries"] = entries
        params["_exits"]   = exits
        qualified[ticker]  = params

    del df, open_s, high_s, low_s, close, vol, atr, entries, exits
    gc.collect()

print(f"\n  Qualified: {list(qualified.keys())}")

# ----------------------------------------------------------------
# Step 5: 今日建議
# ----------------------------------------------------------------
print("\n[Step 5] Generating recommendations...")
recommendations = []

for ticker, params in qualified.items():
    close   = params["_close"]
    open_s  = params["_open"]
    high_s  = params["_high"]
    atr     = params["_atr"]
    entries = params["_entries"]
    exits   = params["_exits"]
    trail, _ = compute_exit_signals(
        open_s, high_s, params["_low"], close, params["_vol"],
        atr, params["atr_mult"], entries
    )

    in_pos_now, last_entry = compute_in_position(entries, exits)
    last_close  = float(close.iloc[-1])
    last_trail  = float(trail.dropna().iloc[-1]) if trail.dropna().shape[0] > 0 else None
    momentum_pct = momentum_scores.get(ticker, np.nan)

    if in_pos_now:
        action       = "HOLD"
        stop_or_rule = f"ATR Stop: {last_trail:.2f}" if last_trail else "—"
    elif entries.iloc[-1]:
        action      = "BUY"
        init_stop   = last_close - float(atr.iloc[-1]) * params["atr_mult"]
        stop_or_rule = f"Init Stop: {init_stop:.2f}"
    else:
        action       = "WAIT"
        stop_or_rule = (
            f"需要: close>{params['trend_ma']}MA, "
            f"vol>{params['vol_mult']}×avgVol, "
            f"強勢K棒"
        )

    recommendations.append({
        "Ticker":        ticker,
        "Action":        action,
        "Last Close":    round(last_close, 2),
        "Momentum 60d":  f"{momentum_pct:.1%}" if not pd.isna(momentum_pct) else "—",
        "Trend MA":      params["trend_ma"],
        "Vol Mult":      params["vol_mult"],
        "Body Ratio":    f"{params['body_ratio']:.0%}",
        "ATR x":         params["atr_mult"],
        "Train PF":      round(params["train_pf"], 2),
        "Test PF":       round(params["test_pf"], 2),
        "Test Trades":   params["test_trades"],
        "Stop / Rule":   stop_or_rule,
        "In Position":   "Yes" if in_pos_now else "No",
        "Last Entry":    str(last_entry.date()) if last_entry else "—",
        "Market":        market_regime,
    })

# ----------------------------------------------------------------
# Step 6: 儲存 + HTML
# ----------------------------------------------------------------
if not recommendations:
    print("\n⚠️  No qualified tickers. Writing empty output.")
    pd.DataFrame().to_csv(RECOMMEND_CSV, index=False)
    with open(HTML_BODY_FILE, "w", encoding="utf-8") as f:
        f.write("<html><body><p>No recommendations today. All tickers failed PF threshold.</p></body></html>")
    sys.exit(0)

rec_df = pd.DataFrame(recommendations)
action_order = {"BUY": 0, "HOLD": 1, "WAIT": 2}
rec_df["_sort"] = rec_df["Action"].map(action_order).fillna(3)
rec_df = rec_df.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
rec_df.to_csv(RECOMMEND_CSV, index=False)

print("\n" + "="*75)
print(rec_df.to_string(index=False))
print("="*75)
print(f"Market: {market_regime.upper()}")
print(f"Saved -> {RECOMMEND_CSV}")

# HTML 信件
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
      <td>{r['Trend MA']}</td>
      <td>{r['Vol Mult']}</td>
      <td>{r['Body Ratio']}</td>
      <td>{r['ATR x']}</td>
      <td>{r['Train PF']}</td>
      <td>{r['Test PF']}</td>
      <td>{r['Test Trades']}</td>
      <td>{r['Stop / Rule']}</td>
      <td>{r['In Position']}</td>
      <td>{r['Last Entry']}</td>
    </tr>"""

regime_color = "#d4edda" if market_regime == "risk-on" else "#f8d7da"
today_str    = datetime.now().strftime("%Y-%m-%d")

html_body = f"""
<html><body style="font-family:Arial;padding:16px">
<h2>📈 Daily Recommendations — {today_str}</h2>
<p>
  Market: <span style="background:{regime_color};padding:3px 10px;border-radius:4px"><b>{market_regime.upper()}</b></span>
  &nbsp;|&nbsp; Top {TOP_N_TICKERS} by {MOMENTUM_WINDOW}d momentum
  &nbsp;|&nbsp; Min Test PF: {MIN_TEST_PF}
</p>
<table border="1" cellpadding="6" cellspacing="0"
       style="border-collapse:collapse;font-size:12px">
  <thead>
    <tr style="background:#343a40;color:white">
      <th>Ticker</th><th>Action</th><th>Last Close</th>
      <th>Momentum</th><th>Trend MA</th><th>Vol Mult</th><th>Body Ratio</th>
      <th>ATR x</th><th>Train PF</th><th>Test PF</th><th>Trades</th>
      <th>Stop / Rule</th><th>In Pos</th><th>Last Entry</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
<p style="font-size:11px;color:gray;margin-top:12px">
  進場條件：Trend MA向上 + 爆量(Vol Mult×) + 強勢K棒(Body Ratio) + 收在高點<br>
  出場條件：ATR trailing stop 或 K線反轉（射擊之星/吞噬/縮量陰線×{WEAK_VOL_DAYS}）<br>
  BUY=<span style="background:#d4edda;padding:2px 6px">green</span>
  HOLD=<span style="background:#fff3cd;padding:2px 6px">yellow</span>
  WAIT=<span style="background:#f8f9fa;padding:2px 6px">gray</span><br>
  ⚠️ 僅供參考，不構成投資建議。
</p>
</body></html>
"""

with open(HTML_BODY_FILE, "w", encoding="utf-8") as f:
    f.write(html_body)

print(f"Saved HTML -> {HTML_BODY_FILE}")
print("Done.")
