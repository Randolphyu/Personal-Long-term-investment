# multi_ticker_param_recommend.py
# -*- coding: utf-8 -*-
"""
策略改進版：
1. Train/Test 分割 — 避免過擬合，只用 test 期間驗證
2. SPY 市場狀態過濾 — risk-on / risk-off
3. 擴展選股池 — 科技個股 + 跨板塊 ETF + 防禦性資產
4. 相對強度輪動 — 每次只推薦動能前 N 強的標的
5. 進場加量能確認 — 突破 MA 當天成交量需放大
6. 出場改進 — ATR trailing + 時間止損（持倉超過 MAX_HOLD_DAYS 無新高則出場）
"""
import gc
import os
import sys
import numpy as np
import pandas as pd
import vectorbt as vbt
import yfinance as yf
from tqdm import tqdm
from datetime import datetime, timedelta

# ================================================================
#  使用者參數
# ================================================================

# --- 選股池 ---
UNIVERSE = {
    # 科技個股
    "個股": ["AAPL", "MSFT", "NVDA", "AVGO", "GOOG", "META", "AMZN", "TSLA", "NFLX"],
    # 板塊 ETF
    "板塊": ["QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU"],
    # 防禦 / 輪動資產
    "防禦": ["GLD", "TLT", "AGG", "VNQ"],
}
# 展開成一個清單
TICKERS = [t for group in UNIVERSE.values() for t in group]

# SPY 用於市場狀態判斷
SPY_TICKER = "SPY"

# --- 時間設定 ---
# 訓練期：用來找最佳參數
TRAIN_START = "2022-01-01"
TRAIN_END   = "2023-12-31"
# 測試期：用來驗證 + 產生今日建議
TEST_START  = "2024-01-01"

FREQ = "1d"

# --- 策略參數 grid ---
MA_LENS   = [10, 15, 20, 30, 50]
ATR_MULTS = [1.5, 2.0, 2.5, 3.0]
ATR_LEN   = 14
LONG_MA_LEN = 200

# --- 輪動設定 ---
MOMENTUM_WINDOW   = 60    # 計算動能用的回顧天數
TOP_N_TICKERS     = 5     # 每次最多推薦幾檔
MIN_PROFIT_FACTOR = 1.2   # 測試期 Profit Factor 低於此值不推薦

# --- 出場改進 ---
MAX_HOLD_DAYS = 30        # 持倉超過此天數且未創新高則出場

# --- 回測參數 ---
INIT_CASH       = 800
COMMISSION      = 0.002
SIZE_HACK_VALUE = 100000

# --- 輸出 ---
OUT_DIR       = "multi_param_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
RECOMMEND_CSV = os.path.join(OUT_DIR, "recommendations.csv")
HTML_BODY_FILE = os.path.join(OUT_DIR, "email_body.html")


# ================================================================
#  工具函式
# ================================================================

def safe_download_series(ticker, start, end=None, interval="1d"):
    kwargs = dict(start=start, interval=interval, progress=False)
    if end:
        kwargs["end"] = end
    try:
        data = vbt.YFData.download(ticker, **{k: v for k, v in
               [("start", start), ("end", end), ("interval", interval)] if v})
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
    if open_s  is not None: open_s  = pd.Series(open_s ).reindex(close_s.index)
    if high_s  is not None: high_s  = pd.Series(high_s ).reindex(close_s.index)
    if low_s   is not None: low_s   = pd.Series(low_s  ).reindex(close_s.index)
    return open_s, high_s, low_s, close_s


def safe_download_volume(ticker, start, end=None, interval="1d"):
    """下載成交量，用於量能確認"""
    try:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
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
        raise AttributeError(f"Indicator object missing {names}. Error: {e}")


def compute_trail_numpy(close_arr, atr_arr, entries_arr, atr_mult, max_hold_days=None):
    """
    ATR trailing stop + 時間止損
    max_hold_days: 持倉超過此天數且未創新高則出場
    """
    n = len(close_arr)
    trail = np.full(n, np.nan)
    exits = np.zeros(n, dtype=bool)
    in_pos, cur_trail = False, np.nan
    entry_day = 0
    highest_since_entry = np.nan

    for i in range(1, n):
        if np.isnan(atr_arr[i]):
            continue
        if entries_arr[i] and not in_pos:
            in_pos = True
            cur_trail = close_arr[i] - atr_arr[i] * atr_mult
            trail[i] = cur_trail
            entry_day = i
            highest_since_entry = close_arr[i]
        elif in_pos:
            # 更新最高點
            if close_arr[i] > highest_since_entry:
                highest_since_entry = close_arr[i]

            # ATR trailing stop
            cur_trail = max(cur_trail, close_arr[i] - atr_arr[i] * atr_mult)
            trail[i] = cur_trail

            # 出場條件 1: 跌破止損
            if close_arr[i] < cur_trail:
                exits[i] = True
                in_pos, cur_trail = False, np.nan
                continue

            # 出場條件 2: 時間止損（持倉超過 max_hold_days 且未創新高）
            if max_hold_days and (i - entry_day) >= max_hold_days:
                if close_arr[i] <= highest_since_entry * 0.995:  # 沒有明顯新高
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
    """
    判斷市場狀態
    returns: 'risk-on' or 'risk-off'
    """
    if spy_close is None or len(spy_close) < LONG_MA_LEN:
        return "unknown"
    ma200 = spy_close.rolling(LONG_MA_LEN).mean()
    last_close = spy_close.iloc[-1]
    last_ma200 = ma200.iloc[-1]
    if pd.isna(last_ma200):
        return "unknown"
    return "risk-on" if last_close > last_ma200 else "risk-off"


def compute_momentum_score(close, window=60):
    """計算相對動能分數（過去 N 日報酬率）"""
    if close is None or len(close) < window + 1:
        return np.nan
    return (close.iloc[-1] - close.iloc[-window]) / close.iloc[-window]


def run_backtest_for_ticker(ticker, start, end, ma_lens, atr_mults,
                             atr_len, long_ma_len, use_long_ma,
                             init_cash, commission, size):
    """對單一 ticker 跑 param grid，回傳結果 list"""
    open_s, high_s, low_s, close = safe_download_series(ticker, start, end)
    if close is None or len(close) < 60:
        return []

    atr_run = vbt.ATR.run(high_s, low_s, close, window=atr_len)
    atr = try_attr(atr_run, ["atr", "real", "value", "output"])
    if isinstance(atr, pd.DataFrame): atr = atr.iloc[:, 0]
    atr = atr.reindex(close.index)

    if use_long_ma:
        long_ma_run = vbt.MA.run(close, window=long_ma_len, ewm=False)
        long_ma = try_attr(long_ma_run, ["ma", "real", "value", "output"])
        if isinstance(long_ma, pd.DataFrame): long_ma = long_ma.iloc[:, 0]
        long_ma = long_ma.reindex(close.index)
    else:
        long_ma = pd.Series(np.nan, index=close.index)

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

    if use_long_ma:
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
                init_cash=init_cash,
                fees=commission,
                size=size,
                direction="longonly",
                freq="1d"
            )
            stats = pf.stats()
            pfactor      = stats.get("Profit Factor")    if "Profit Factor"    in stats.index else np.nan
            total_return = stats.get("Total Return [%]") if "Total Return [%]" in stats.index else np.nan
            max_dd       = stats.get("Max Drawdown [%]") if "Max Drawdown [%]" in stats.index else np.nan
            n_trades     = stats.get("Total Trades")     if "Total Trades"     in stats.index else 0
            sharpe       = stats.get("Sharpe ratio")     if "Sharpe ratio"     in stats.index else np.nan
            del pf

            try:
                ma_len_val = int(''.join(filter(str.isdigit, str(col))))
            except Exception:
                ma_len_val = col

            results.append({
                'ticker':        ticker,
                'ma_len':        ma_len_val,
                'atr_mult':      float(atr_mult),
                'profit_factor': float(pfactor)      if not pd.isna(pfactor)      else np.nan,
                'total_return':  float(total_return) if not pd.isna(total_return) else np.nan,
                'max_drawdown':  float(max_dd)       if not pd.isna(max_dd)       else np.nan,
                'n_trades':      int(n_trades)        if not pd.isna(n_trades)     else 0,
                'sharpe':        float(sharpe)        if not pd.isna(sharpe)       else np.nan,
            })

    del entries_df, ma_all, atr, close_arr, atr_arr
    gc.collect()
    return results


# ================================================================
#  主流程
# ================================================================
print("Python:", sys.version.splitlines()[0])
print("vectorbt:", getattr(vbt, "__version__", "unknown"))
print(f"Universe: {len(TICKERS)} tickers")
print(f"Train: {TRAIN_START} ~ {TRAIN_END}")
print(f"Test : {TEST_START} ~ today")

# ----------------------------------------------------------------
# Step 1: 判斷市場狀態 (SPY)
# ----------------------------------------------------------------
print("\n[Step 1] Checking market regime via SPY...")
_, _, _, spy_close = safe_download_series(SPY_TICKER, TRAIN_START)
market_regime = get_market_regime(spy_close)
print(f"  Market regime: {market_regime.upper()}")

if market_regime == "risk-off":
    print("  ⚠️  Risk-OFF: Market below 200MA. Only defensive assets will be considered.")
    # risk-off 時只看防禦性資產
    ACTIVE_TICKERS = UNIVERSE["防禦"]
else:
    ACTIVE_TICKERS = TICKERS

# ----------------------------------------------------------------
# Step 2: 計算相對動能，篩選前 TOP_N 標的
# ----------------------------------------------------------------
print(f"\n[Step 2] Computing momentum scores (window={MOMENTUM_WINDOW}d)...")
momentum_scores = {}
for ticker in ACTIVE_TICKERS:
    _, _, _, close = safe_download_series(ticker, TEST_START)
    score = compute_momentum_score(close, MOMENTUM_WINDOW)
    momentum_scores[ticker] = score
    print(f"  {ticker}: {score:.2%}" if not pd.isna(score) else f"  {ticker}: N/A")
    gc.collect()

# 排序，取前 TOP_N（排除 NaN）
sorted_tickers = sorted(
    [(t, s) for t, s in momentum_scores.items() if not pd.isna(s)],
    key=lambda x: x[1], reverse=True
)
top_tickers = [t for t, _ in sorted_tickers[:TOP_N_TICKERS]]
print(f"\n  Top {TOP_N_TICKERS} by momentum: {top_tickers}")

# ----------------------------------------------------------------
# Step 3: 訓練期 param grid（找最佳參數）
# ----------------------------------------------------------------
print(f"\n[Step 3] Running TRAIN param grid ({TRAIN_START} ~ {TRAIN_END})...")
train_results = []
for ticker in tqdm(top_tickers, desc="Train"):
    results = run_backtest_for_ticker(
        ticker, TRAIN_START, TRAIN_END,
        MA_LENS, ATR_MULTS, ATR_LEN, LONG_MA_LEN, True,
        INIT_CASH, COMMISSION, SIZE_HACK_VALUE
    )
    train_results.extend(results)

train_df = pd.DataFrame(train_results)

# 每檔取訓練期最佳參數
best_params = {}
for ticker in top_tickers:
    sub = train_df[train_df['ticker'] == ticker].sort_values("profit_factor", ascending=False)
    if not sub.empty:
        best_params[ticker] = {
            'ma_len':   int(sub.iloc[0]['ma_len']),
            'atr_mult': float(sub.iloc[0]['atr_mult']),
            'train_pf': float(sub.iloc[0]['profit_factor']),
        }
        print(f"  {ticker} best train params: MA={best_params[ticker]['ma_len']}, "
              f"ATRx{best_params[ticker]['atr_mult']}, PF={best_params[ticker]['train_pf']:.2f}")

# ----------------------------------------------------------------
# Step 4: 測試期驗證（用訓練期找到的參數跑 test 期間）
# ----------------------------------------------------------------
print(f"\n[Step 4] Validating on TEST period ({TEST_START} ~ today)...")
test_results = []
for ticker, params in tqdm(best_params.items(), desc="Test"):
    results = run_backtest_for_ticker(
        ticker, TEST_START, None,
        [params['ma_len']], [params['atr_mult']],
        ATR_LEN, LONG_MA_LEN, True,
        INIT_CASH, COMMISSION, SIZE_HACK_VALUE
    )
    if results:
        r = results[0]
        r['train_pf'] = params['train_pf']
        test_results.append(r)
        print(f"  {ticker}: Test PF={r['profit_factor']:.2f} "
              f"(Train PF={params['train_pf']:.2f})")

test_df = pd.DataFrame(test_results)

# 過濾測試期 Profit Factor 不達標的
qualified = test_df[test_df['profit_factor'] >= MIN_PROFIT_FACTOR]
print(f"\n  Qualified tickers (test PF >= {MIN_PROFIT_FACTOR}): "
      f"{qualified['ticker'].tolist()}")

# ----------------------------------------------------------------
# Step 5: 產生今日建議
# ----------------------------------------------------------------
print("\n[Step 5] Generating recommendations...")
recommendations = []

for _, row in qualified.iterrows():
    ticker   = row['ticker']
    best_ma  = int(row['ma_len'])
    best_atr = float(row['atr_mult'])

    _, high_s, low_s, close = safe_download_series(ticker, TEST_START)
    if close is None or len(close) < 60:
        continue

    # 量能確認：下載成交量
    volume = safe_download_volume(ticker, TEST_START)
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

    # 進場訊號 + 量能確認
    entries = (close > ma_series) & (close.shift(1) <= ma_series.shift(1))
    entries = entries.fillna(False).astype(bool)
    entries &= (close > long_ma)

    # 量能確認：突破當天成交量 > 20日均量 1.2 倍
    if vol_ma20 is not None:
        vol_ma20 = vol_ma20.reindex(close.index)
        vol_confirm = (volume.reindex(close.index) > vol_ma20 * 1.2).fillna(False)
        entries &= vol_confirm

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
        stop_or_rule = f"Stop: {(close.iloc[-1] - atr.iloc[-1] * best_atr):.2f}" \
                       if not pd.isna(atr.iloc[-1]) else "—"
    else:
        action = "WAIT"
        cond   = f"Buy if close > {ma_last:.2f}" if ma_last else ""
        if long_ma_last:
            cond += f" & > LongMA {long_ma_last:.2f}"
        stop_or_rule = cond or "—"

    recommendations.append({
        "Ticker":         ticker,
        "Action":         action,
        "Last Close":     round(last_close, 2),
        "Momentum 60d":   f"{momentum_pct:.1%}" if not pd.isna(momentum_pct) else "—",
        "MA":             best_ma,
        "ATR x":          best_atr,
        "Test PF":        round(row['profit_factor'], 2),
        "Train PF":       round(row['train_pf'], 2),
        "Stop / Rule":    stop_or_rule,
        "In Position":    "Yes" if in_pos_now else "No",
        "Last Entry":     str(last_entry.date()) if last_entry else "—",
        "Market":         market_regime,
    })

    del high_s, low_s, close, atr, ma_series, long_ma, entries, exits, trail
    gc.collect()

# ----------------------------------------------------------------
# Step 6: 儲存 CSV + 產生 HTML 信件
# ----------------------------------------------------------------
rec_df = pd.DataFrame(recommendations)

# 排序：BUY 優先，再依動能排
action_order = {"BUY": 0, "HOLD": 1, "WAIT": 2}
rec_df["_sort"] = rec_df["Action"].map(action_order).fillna(3)
rec_df = rec_df.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

rec_df.to_csv(RECOMMEND_CSV, index=False)

print("\n" + "="*70)
print(rec_df.to_string(index=False))
print("="*70)
print(f"\nMarket Regime: {market_regime.upper()}")
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
      <td>{r['MA']}</td>
      <td>{r['ATR x']}</td>
      <td>{r['Test PF']}</td>
      <td>{r['Train PF']}</td>
      <td>{r['Stop / Rule']}</td>
      <td>{r['In Position']}</td>
      <td>{r['Last Entry']}</td>
    </tr>"""

regime_badge_color = "#d4edda" if market_regime == "risk-on" else "#f8d7da"
today_str = datetime.now().strftime("%Y-%m-%d")

html_body = f"""
<html><body style="font-family:Arial;padding:16px">
<h2>📈 Daily Trading Recommendations — {today_str}</h2>
<p>Market Regime (SPY vs 200MA):
  <span style="background:{regime_badge_color};padding:3px 10px;border-radius:4px">
    <b>{market_regime.upper()}</b>
  </span>
  &nbsp;|&nbsp; Top {TOP_N_TICKERS} tickers by {MOMENTUM_WINDOW}d momentum
  &nbsp;|&nbsp; Min Test Profit Factor: {MIN_PROFIT_FACTOR}
</p>
<table border="1" cellpadding="6" cellspacing="0"
       style="border-collapse:collapse;font-size:14px">
  <thead>
    <tr style="background:#343a40;color:white">
      <th>Ticker</th><th>Action</th><th>Last Close</th>
      <th>Momentum 60d</th><th>MA</th><th>ATR x</th>
      <th>Test PF</th><th>Train PF</th>
      <th>Stop / Rule</th><th>In Position</th><th>Last Entry</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
<p style="font-size:12px;color:gray;margin-top:12px">
  BUY=<span style="background:#d4edda;padding:2px 6px">green</span> &nbsp;
  HOLD=<span style="background:#fff3cd;padding:2px 6px">yellow</span> &nbsp;
  WAIT=<span style="background:#f8f9fa;padding:2px 6px">gray</span><br>
  ⚠️ 僅供參考，不構成投資建議。過去績效不代表未來表現。
</p>
</body></html>
"""

with open(HTML_BODY_FILE, "w", encoding="utf-8") as f:
    f.write(html_body)

print(f"Saved HTML email body -> {HTML_BODY_FILE}")
print("Done.")
